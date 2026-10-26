"""履歴・独立比較・保存・中断・上限を、ネットワークなしで検証する。"""

from types import SimpleNamespace

import pytest
from google.genai import types
from app.ask import Conversation
from app.study import (StudySession, MODELS, START, END,
                       MAX_LOG_BYTES, RESERVE_BYTES)
from app.study_cli import menu


class FakeClient:
    def __init__(self, action=None):
        self.models = self
        self.requests = []
        self.closed = False
        self.action = action

    def generate_content(self, **kwargs):
        self.requests.append(kwargs)
        if self.action:
            return self.action()
        return SimpleNamespace(
            text="カワウソ。テスト用回答です。",
            usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=7,
                                           thoughts_token_count=3, total_token_count=20),
            candidates=[SimpleNamespace(finish_reason=types.FinishReason.STOP)],
        )

    def close(self):
        self.closed = True


class ApiFailure(Exception):
    code = 429


def setup(tmp_path, action=None, **kwargs):
    clients = []
    def factory(**params):
        client = FakeClient(action)
        clients.append(client)
        return Conversation(client=client, **params)
    answer = tmp_path / "lectures/01/answer.md"
    answer.parent.mkdir(parents=True)
    answer.write_text("# 自分の解答\n手書きは消さない\n" + START + "\n" + END + "\n末尾メモ\n")
    session = StudySession("fake-secret-for-tests", root=tmp_path, factory=factory,
                           interval=0, **kwargs)
    return session, clients, answer


def test_conversation_sends_user_model_user_and_reset():
    client = FakeClient()
    conversation = Conversation(client=client)
    first = conversation.send("動物を覚えて")
    second = conversation.send("何の動物？")
    assert [c.role for c in client.requests[1]["contents"]] == ["user", "model", "user"]
    assert first.history_messages == 0 and second.history_messages == 2
    assert second.total_tokens == 20 and second.thinking_tokens == 3
    assert client.requests[0]["config"].max_output_tokens == 2048
    assert client.requests[0]["config"].automatic_function_calling.disable is True
    conversation.reset()
    conversation.send("新しい話")
    assert len(client.requests[2]["contents"]) == 1


def test_failed_or_empty_response_does_not_extend_history():
    client = FakeClient(lambda: (_ for _ in ()).throw(ApiFailure("rate limit")))
    conversation = Conversation(client=client)
    with pytest.raises(ApiFailure):
        conversation.send("質問")
    assert conversation.history == []
    client.action = lambda: SimpleNamespace(text=None, usage_metadata=None, candidates=[])
    reply = conversation.send("質問")
    assert reply.input_tokens is None and reply.text == ""
    assert conversation.history == []


def test_compare_12_calls_are_isolated_and_report_uses_real_metadata(tmp_path):
    session, clients, answer = setup(tmp_path)
    session.compare(lambda _: None)
    events = session.journal.read()
    assert len(events) == session.calls == 12
    assert len(clients) == 12 and all(c.closed for c in clients)
    assert all(len(c.requests[0]["contents"]) == 1 for c in clients)
    report = answer.read_text()
    assert "12回の比較実験を実施済み" in report
    assert "10.00 (n=3)" in report and "20.00 (n=3)" in report
    assert "手書きは消さない" in report and report.endswith("末尾メモ\n")
    assert "429は 0 件" in report
    assert "限界まで呼び出した場合の挙動は判断できません" in report


def test_error_stops_comparison_no_retry_and_partial_is_not_complete(tmp_path):
    session, clients, answer = setup(tmp_path, action=lambda: (_ for _ in ()).throw(ApiFailure("429")))
    session.compare(lambda _: None)
    assert session.calls == len(clients) == 1 and len(clients[0].requests) == 1
    assert session.halted
    assert "未完了" in answer.read_text() and "429は 1 件" in answer.read_text()
    with pytest.raises(RuntimeError, match="停止中"):
        session.call(MODELS[0], "続けて")
    assert session.calls == 1


def test_keyboard_interrupt_saves_attempt_and_closes_client(tmp_path):
    session, clients, answer = setup(tmp_path, action=lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        session.compare(lambda _: None)
    assert session.journal.read()[0]["status"] == "interrupted"
    assert clients[0].closed and "未完了" in answer.read_text()


def test_budget_space_model_and_secret_limits_prevent_network(tmp_path):
    session, clients, answer = setup(tmp_path, max_calls=1)
    with pytest.raises(RuntimeError, match="12回"):
        session.compare()
    with pytest.raises(ValueError, match="2モデル"):
        session.call("unverified-model", "質問")
    with pytest.raises(ValueError, match="APIキー"):
        session.call(MODELS[0], "fake-secret-for-tests")
    session.journal.max_bytes = RESERVE_BYTES - 1
    # 空ファイルにも予約領域が必要。
    session.journal.path.touch()
    with pytest.raises(RuntimeError, match="20MB"):
        session.call(MODELS[0], "質問")
    assert clients == [] and session.calls == 0
    session.journal.max_bytes = MAX_LOG_BYTES
    session.call(MODELS[0], "質問")
    with pytest.raises(RuntimeError, match="上限"):
        session.call(MODELS[0], "もう一度")
    assert len(clients[0].requests) == 1


def test_context_limit_does_not_consume_api_budget_and_reset_recovers(tmp_path):
    session, clients, _ = setup(tmp_path)
    session.call(MODELS[0], "質問")
    conversation = session.conversations[MODELS[0]]
    conversation.history[1].parts[0].text = "あ" * 12000
    with pytest.raises(ValueError, match="/new"):
        session.call(MODELS[0], "続き")
    assert session.calls == 1 and not session.halted
    session.reset()
    session.call(MODELS[0], "新しい会話")
    assert session.calls == 2 and len(clients) == 2


def test_secret_removed_from_logs_reports_and_export_and_output(tmp_path):
    export = tmp_path / "export/answer.md"
    session, clients, answer = setup(tmp_path, export_path=export)
    clients_response = FakeClient()
    base = clients_response.generate_content()
    base.text = "fake-secret-for-tests と " + "AIza" + "a" * 35
    session.factory = lambda **params: Conversation(client=FakeClient(lambda: base), **params)
    event = session.call(MODELS[0], "普通の質問")
    session.note("fake-secret-for-tests を含むメモ")
    for content in [session.journal.path.read_text(), answer.read_text(), export.read_text(), event["answer"]]:
        assert "fake-secret-for-tests" not in content
        assert "AIza" + "a" * 35 not in content
        assert "[APIキー非表示]" in content


def test_missing_markers_preserve_manual_file_and_write_fallback(tmp_path):
    session, _, answer = setup(tmp_path)
    answer.write_text("自分で書き直した解答")
    path = session.report()
    assert answer.read_text() == "自分で書き直した解答"
    assert path.name == "answer.generated.md" and "未完了" in path.read_text()


def test_missing_usage_and_truncated_response_not_faked_as_success(tmp_path):
    base = FakeClient().generate_content()
    base.usage_metadata = None
    base.candidates[0].finish_reason = types.FinishReason.MAX_TOKENS
    session, _, answer = setup(tmp_path, action=lambda: base)
    session.compare(lambda _: None)
    events = session.journal.read()
    assert all(e["status"] == "truncated" and e["input_tokens"] is None for e in events)
    assert "未完了" in answer.read_text()
    assert "10.00" not in answer.read_text()
    assert "None" not in answer.read_text() and "未取得" in answer.read_text()


def test_sdk_retry_disabled_and_timeout_bounded(monkeypatch):
    calls = []
    monkeypatch.setattr("app.ask.genai.Client", lambda **kw: calls.append(kw) or FakeClient())
    Conversation(api_key="fake-secret-for-tests")
    assert calls[0]["http_options"].retry_options.attempts == 1
    assert calls[0]["http_options"].timeout == 60000
    assert calls[0]["vertexai"] is False


def test_wizard_all_levels_and_notes_without_real_api(tmp_path, monkeypatch):
    session, clients, answer = setup(tmp_path)
    answers = iter(["1", "3", "回答の表現には違いがあった", "0"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    menu(session)
    assert session.calls == 14
    assert len(session.journal.read()) == 15
    assert [c.role for c in clients[-1].requests[-1]["contents"]] == ["user", "model", "user"]
    assert "履歴付きの成功記録 1 件" in answer.read_text()
    assert "回答の表現には違いがあった" in answer.read_text()


def test_rate_limit_spacing_excludes_wait_from_api_timing(tmp_path):
    clock = [0.0]
    waits = []
    def pause(seconds):
        waits.append(seconds)
        clock[0] += seconds
    session, _, _ = setup(tmp_path, clock=lambda: clock[0], sleep_fn=pause)
    session.interval = 15
    session.call(MODELS[0], "質問1")
    clock[0] += 4
    event = session.call(MODELS[0], "質問2")
    assert waits == [11]
    assert event["elapsed_seconds"] < 1


def test_json_escaping_still_saves_bounded_record_and_marks_truncation(tmp_path):
    session, clients, answer = setup(tmp_path)
    base = FakeClient().generate_content()
    base.text = '"' * 10000
    session.factory = lambda **params: Conversation(client=FakeClient(lambda: base), **params)
    event = session.call(MODELS[0], '"' * 2000)
    assert len(session.journal.path.read_text()) <= 20000
    assert event["status"] == "truncated" and event["answer_truncated"] is True
    assert len(session.journal.read()) == 1


def test_new_journal_checks_capacity_before_first_call(tmp_path):
    session, clients, _ = setup(tmp_path)
    session.journal.max_bytes = 1
    with pytest.raises(RuntimeError, match="20MB"):
        session.call(MODELS[0], "質問")
    assert clients == []


def test_manual_resume_skips_six_successes_keeps_error_and_finishes_memory(tmp_path):
    from app.study_cli import run_all
    attempts = [0]
    class Busy(Exception):
        code = 503
    def response():
        attempts[0] += 1
        if attempts[0] == 7:
            raise Busy("high demand")
        return FakeClient().generate_content()
    first, clients, answer = setup(tmp_path, action=response)
    first.compare(lambda _: None)
    assert first.calls == 7 and first.halted
    with pytest.raises(RuntimeError, match="開き直して"):
        first.compare(lambda _: None, resume=True)
    assert attempts[0] == 7
    second = StudySession("fake-secret-for-tests", root=tmp_path, factory=first.factory, interval=0)
    experiment, pending = second.comparison_plan(resume=True)
    assert len(pending) == 6 and all(model == MODELS[1] for model, _, _ in pending)
    run_all(second, resume=True)
    assert second.calls == 8  # 残り比較6回＋会話2回。
    records = second.journal.read()
    assert len(records) == 15 and sum(e["status"] == "error" for e in records) == 1
    comparison = [e for e in records if e["mode"] == "comparison" and e["status"] == "ok"]
    assert len({(e["model"], e["question"], e["repeat"]) for e in comparison}) == 12
    assert "12回の比較実験を実施済み" in answer.read_text()
    assert "履歴付きの成功記録 1 件" in answer.read_text()
    assert "503（サービス利用不可）は 1 件" in answer.read_text()
    third = StudySession("fake-secret-for-tests", root=tmp_path, factory=first.factory, interval=0)
    run_all(third, resume=True)
    assert third.calls == 0 and attempts[0] == 15


def test_resume_without_saved_experiment_does_not_call_api(tmp_path):
    session, clients, _ = setup(tmp_path)
    with pytest.raises(RuntimeError, match="再開する比較実験がありません"):
        session.compare(lambda _: None, resume=True)
    assert clients == [] and session.calls == 0


def test_resume_after_explicit_model_change_reuses_lite_successes(tmp_path):
    from app.study import QUESTIONS, now
    from app.study_cli import run_all
    session, clients, answer = setup(tmp_path)
    assert MODELS == ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite")
    for question in QUESTIONS:
        for repeat in range(1, 4):
            session.call(MODELS[0], question, mode="comparison", experiment="before-change", repeat=repeat)
    session.journal.append({
        "kind": "turn", "time": now(), "model": "gemini-3.8-flash", "mode": "comparison",
        "experiment": "before-change", "question": QUESTIONS[0], "repeat": 1,
        "status": "error", "error_code": 504, "error": "Deadline exceeded",
        "history_messages": 0, "answer": "",
    })
    experiment, pending = session.comparison_plan(resume=True)
    assert experiment == "before-change" and len(pending) == 6
    assert all(model == "gemini-3.1-flash-lite" for model, _, _ in pending)
    run_all(session, resume=True)
    assert session.calls == 14  # 先に6回、切替後は比較6回＋会話2回。
    report = answer.read_text()
    assert "12回の比較実験を実施済み" in report
    assert "504（タイムアウト）は 1 件" in report
    assert sum(e.get("error_code") == 504 for e in session.journal.read()) == 1
