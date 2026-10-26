"""第1回の記録・比較・提出用下書き。集計にはLLMを呼ばない。"""

from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from difflib import SequenceMatcher
import fcntl
import html
import json
import os
from pathlib import Path
import re
from statistics import mean
import tempfile
from time import monotonic, sleep
from uuid import uuid4
from zoneinfo import ZoneInfo

from app.ask import Conversation, DEFAULT_QUESTION

ROOT = Path(__file__).resolve().parents[1]
FREE_TIER = json.loads((ROOT / "app/free_tier.json").read_text())
MODELS = tuple(FREE_TIER["comparison_models"])
QUESTIONS = ("LoRAとは何ですか？", DEFAULT_QUESTION)
START = "<!-- AIE:AUTO:START -->"
END = "<!-- AIE:AUTO:END -->"
MAX_LOG_BYTES = 20 * 1024 * 1024
MAX_CALLS = 30
RESERVE_BYTES = 80000


def now():
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds")


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        f.write(text)
    os.replace(temp, path)


class Journal:
    def __init__(self, path, secret="", max_bytes=MAX_LOG_BYTES):
        self.path = Path(path)
        self.secret = secret
        self.max_bytes = max_bytes
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    def clean(self, text):
        text = str(text)
        if self.secret:
            text = text.replace(self.secret, "[APIキー非表示]")
        return re.sub(r"AIza[\w-]{20,}", "[APIキー非表示]", text)

    def check_space(self):
        size = self.path.stat().st_size if self.path.exists() else 0
        if size + RESERVE_BYTES > self.max_bytes:
            raise RuntimeError("学習ログが20MB上限に近づいたため、APIを呼ばずに停止しました。")

    def append(self, event):
        safe = {k: self.clean(v) if isinstance(v, str) else v for k, v in event.items()}
        line = json.dumps(safe, ensure_ascii=False) + "\n"
        while len(line) > 20000 and safe.get("answer"):
            # 引用符・改行のJSONエスケープ分も含めて上限に収める。
            safe["answer"] = safe["answer"][:len(safe["answer"]) // 2]
            safe["answer_truncated"] = True
            if safe.get("status") == "ok":
                safe["status"] = "truncated"
            line = json.dumps(safe, ensure_ascii=False) + "\n"
        if len(line) > 20000:
            raise ValueError("1件の学習ログが2万文字を超えました。")
        with self.path.open("a", encoding="utf-8") as f:
            os.chmod(self.path, 0o600)
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.seek(0, 2)
            if f.tell() + len(line.encode()) > self.max_bytes:
                raise RuntimeError("学習ログの上限です。")
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        return safe

    def read(self):
        if not self.path.exists():
            return []
        events = []
        with self.path.open(encoding="utf-8") as file:
            fcntl.flock(file.fileno(), fcntl.LOCK_SH)
            lines = file.read().splitlines()
        for n, line in enumerate(lines, 1):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                raise RuntimeError(f"学習ログの{n}行目が壊れています。原本を残して確認してください。")
        return events


def cell(value):
    if value is None:
        return "未取得"
    return html.escape(str(value)).replace("|", "&#124;").replace("\n", " ")


def avg(rows, name):
    values = [r[name] for r in rows if r.get(name) is not None]
    return f"{mean(values):.2f} (n={len(values)})" if values else "未取得"


def report_body(events):
    turns = [e for e in events if e["kind"] == "turn"]
    comparisons = [e for e in turns if e["mode"] == "comparison"]
    conversations = [e for e in turns if e["mode"] == "conversation"]
    valid = [e for e in comparisons if e["status"] == "ok"]
    groups = defaultdict(list)
    experiments = defaultdict(list)
    for e in comparisons:
        experiments[e["experiment"]].append(e)
    complete = [key for key, rows in experiments.items() if all(
        sum(e["model"] == m and e["question"] == q and e["status"] == "ok" for e in rows) == 3
        for m in MODELS for q in QUESTIONS
    )]
    for e in valid:
        groups[(e["model"], e["question"])].append(e)
    memory = [e for e in conversations if e["status"] == "ok" and e["history_messages"] > 0]
    lines = [
        "## 自動集計（実際に保存された記録のみ）", "",
        f"更新日時: {now()}", "",
        "| 課題 | 現在の状態 |", "|---|---|",
        f"| 基礎 | {'12回の比較実験を実施済み' if complete else '未完了（2モデル×2問×3回の成功記録が必要）'} |",
        f"| 実装 | 会話履歴の保持を実装済み。履歴付きの成功記録 {len(memory)} 件。回答内容の妥当性は考察欄で確認する |",
        f"| 発展 | 比較実験の記録 {len(comparisons)} 件、本文を取得できた成功記録 {len(valid)} 件。応答時間・トークン数は下表 |", "",
        "### やったこと・条件", "",
        "- 基礎・発展: 2モデル×2質問×各3回。各回は履歴なしの独立した質問。",
        "- 現在の比較モデル: " + " / ".join(MODELS),
        "- " + FREE_TIER["comparison_change"],
        "- 実装: app/ask.py の Conversation に user/model の履歴を保持し、次の質問へ渡す。失敗したターンは追加しない。",
        "- 出力上限2048トークン。temperatureは指定せずAPI既定値。同一条件でも出力が変わる可能性がある。",
        "- 応答時間はAPI呼び出し全体の経過秒数。待機時間とファイル保存時間は含まない。",
        "- 思考トークンと合計トークンも記録。未取得の値は0とせず未取得と表示する。",
        "- 集計・下書き作成はPythonで行い、追加のLLM呼び出しはしない。", "",
        "### 基礎: 回答の揺れ（文字列上の比較）", "",
    ]
    if not valid:
        lines.append("比較実験はまだ記録されていません。実行結果は作成していません。")
    for (model, question), rows in groups.items():
        texts = [re.sub(r"\s+", " ", e["answer"]).strip() for e in rows]
        recent = [text[:2000] for text in texts[-3:]]
        pairs = [SequenceMatcher(None, a, b).ratio()
                 for i, a in enumerate(recent) for b in recent[i+1:]]
        similarity = f"、直近3件以内・先頭2000文字の平均文字列類似度 {mean(pairs):.3f}" if pairs else ""
        lines.append(f"- {cell(model)} / {cell(question)}: {len(rows)}回答、異なる本文 {len(set(texts))}種類{similarity}。")
    lines += [
        "", "文字列の違い・類似度は、内容の正しさや品質を判定する指標ではありません。", "",
        "講座固有の質問の参照正解（質問時にはモデルに渡さない）:",
        "> " + cell(next(t["definition"] for t in json.loads((ROOT / "data/glossary.json").read_text())["terms"] if t["id"] == "homework-levels")),
        "", "### 発展: モデル・質問別の実測平均", "",
        "| モデル | 質問 | 成功n | 応答秒 | 入力 | 出力 | 思考 | 合計トークン |",
        "|---|---|---:|---|---|---|---|---|",
    ]
    for (model, question), rows in groups.items():
        lines.append("| " + " | ".join([
            cell(model), cell(question), str(len(rows)), avg(rows, "elapsed_seconds"),
            avg(rows, "input_tokens"), avg(rows, "output_tokens"),
            avg(rows, "thinking_tokens"), avg(rows, "total_tokens"),
        ]) + " |")
    errors = [e for e in turns if e["status"] not in ("ok", "truncated")]
    limited = [e for e in errors if e.get("error_code") == 429]
    unavailable = [e for e in errors if e.get("error_code") == 503]
    deadlines = [e for e in errors if e.get("error_code") == 504]
    lines += ["", "### 無料枠とレート制限", "",
        f"- {FREE_TIER['checked_on']} に本人のAI Studioで無料枠を確認したスナップショット。",
        *[f"- {model}: RPM {FREE_TIER['models'][model]['rpm']}、TPM {FREE_TIER['models'][model]['tpm']}、RPD {FREE_TIER['models'][model]['rpd']}。" for model in MODELS],
        "- 連続呼び出しは開始間隔を最低15秒に制限し、最大30回/起動、比較は1回の実行で最大12回。手動再開では成功済みを除く残りだけ呼び出す。",
        f"- 記録された429は {len(limited)} 件。" + ("制限の発生を観測しました。" if limited else "この記録だけでは、限界まで呼び出した場合の挙動は判断できません。"),
        f"- 記録された503（サービス利用不可）は {len(unavailable)} 件。429のレート制限とは区別して記録します。",
        f"- 記録された504（タイムアウト）は {len(deadlines)} 件。401の認証拒否とは区別して記録します。",
        "- RPM/RPDは呼び出し回数、TPMは入力トークン数の制約。会話履歴が長いほど入力が増えます。",
        "- 実測の応答時間とTPM/RPM/RPDの制限は別です。他の端末・キーからの利用も同一プロジェクトの枠を共有します。",
        "- エラー時の自動再試行や別モデルへの自動切替は行いません。上限は変わるのでAI Studioで再確認します。",
        "- 公式資料: https://ai.google.dev/gemini-api/docs/rate-limits", "",
        "<details>", "<summary>実行記録（直近50件の本文・測定値）</summary>", "",
    ]
    for e in turns[-50:]:
        lines += [f"#### {cell(e['time'])} / {cell(e['model'])} / {cell(e['mode'])}",
            f"- 状態: {cell(e['status'])} / 履歴メッセージ数: {e['history_messages']}",
            f"- 応答秒: {cell(e.get('elapsed_seconds'))} / 入力: {cell(e.get('input_tokens'))} / 出力: {cell(e.get('output_tokens'))} / 思考: {cell(e.get('thinking_tokens'))} / 合計: {cell(e.get('total_tokens'))}",
            f"- finish_reason: {cell(e.get('finish_reason', '未取得'))}",
            "", "質問:", "> " + cell(e['question']), "", "回答:",
            *["> " + html.escape(line) for line in e.get("answer", "（回答なし）").splitlines()], ""]
        if e.get("error"):
            lines.append("エラー: " + cell(e["error"]))
        if e.get("answer_truncated"):
            lines.append("※保存文字数の上限により本文の後半を省略しています。")
    notes = [e for e in events if e["kind"] == "note"]
    lines += ["", "</details>", "", "### 観察・考察メモ", ""]
    lines += [f"- {cell(e['text'])}" for e in notes] or ["まだ入力されていません。正しさの比較、難しかった点、自分の気づきを追記してください。"]
    return "\n".join(line.rstrip() for line in lines) + "\n"


def write_report(journal, answer_path, export_path=None):
    answer_path = Path(answer_path)
    content = journal.clean(report_body(journal.read()))
    existing = answer_path.read_text() if answer_path.exists() else "# 第1回 解答\n\n" + START + "\n" + END + "\n"
    if existing.count(START) == 1 and existing.count(END) == 1 and existing.index(START) < existing.index(END):
        before, rest = existing.split(START)
        _, after = rest.split(END)
        output = before + START + "\n" + content + END + after
    else:
        # 自動生成区間がなくなった場合は手書きの原本を上書きしない。
        answer_path = answer_path.with_name("answer.generated.md")
        output = "# 第1回 解答（自動集計の別ファイル）\n\n" + content
    output = journal.clean(output)
    atomic_write(answer_path, output)
    if export_path:
        atomic_write(Path(export_path), output)
    return answer_path


class StudySession:
    def __init__(self, secret, *, root=ROOT, export_path=None, factory=Conversation,
                 interval=15, sleep_fn=sleep, clock=monotonic, max_calls=MAX_CALLS):
        self.secret = secret
        self.root = Path(root)
        self.journal = Journal(self.root / ".logs/lec01/events.jsonl", secret)
        self.export_path = export_path
        self.factory = factory
        self.interval, self.sleep, self.clock = interval, sleep_fn, clock
        self.max_calls, self.calls = max_calls, 0
        self.run = uuid4().hex
        self.last_start = None
        self.halted = False
        self.conversations = {}

    def report(self):
        return write_report(self.journal, self.root / "lectures/01/answer.md", self.export_path)

    def note(self, text):
        if not text.strip() or len(text) > 2000:
            raise ValueError("メモは1〜2000文字にしてください。")
        self.journal.append({"kind": "note", "time": now(), "text": text, "run": self.run})
        self.report()

    def reset(self):
        for conversation in self.conversations.values():
            conversation.close()
        self.conversations.clear()

    def call(self, model, question, *, mode="conversation", experiment="", repeat=1):
        if model not in MODELS:
            raise ValueError("無料枠を確認した2モデル以外は、この起動ツールでは使用しません。")
        if self.halted:
            raise RuntimeError("前のAPIエラーにより停止中です。自動再試行しません。")
        if self.calls >= self.max_calls:
            raise RuntimeError("この起動のAPI呼び出し上限30回に達しました。")
        if not question.strip() or len(question) > 2000:
            raise ValueError("質問は1〜2000文字にしてください。")
        if len(self.secret) > 8 and self.secret in question:
            raise ValueError("質問にAPIキーが含まれているため送信しませんでした。")
        self.journal.check_space()
        conversation = self.conversations.get(model) if mode == "conversation" else None
        if conversation is None:
            conversation = self.factory(model=model, api_key=self.secret)
            if mode == "conversation":
                self.conversations[model] = conversation
        try:
            conversation.validate(question)
        except ValueError:
            if mode != "conversation":
                conversation.close()
            raise
        if self.last_start is not None:
            wait = self.interval - (self.clock() - self.last_start)
            if wait > 0:
                self.sleep(wait)
        self.last_start = self.clock()
        self.calls += 1
        event = {"kind": "turn", "time": now(), "run": self.run,
                 "model": model, "question": question, "mode": mode,
                 "experiment": experiment, "repeat": repeat,
                 "history_messages": len(conversation.history), "answer": ""}
        try:
            reply = conversation.send(question)
            event.update(asdict(reply))
            safe_text = self.journal.clean(event.pop("text"))
            event["answer"] = safe_text[:10000]
            event["answer_truncated"] = len(safe_text) > 10000
            event["status"] = ("ok" if reply.text and reply.finish_reason == "STOP"
                               and not event["answer_truncated"] else "truncated" if reply.text else "empty")
            if event["status"] == "empty":
                self.halted = True
        except (KeyboardInterrupt, Exception) as error:
            event.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "error",
                         error_code=getattr(error, "code", None),
                         elapsed_seconds=round(self.clock() - self.last_start, 6),
                         error=self.journal.clean(str(error))[:1000])
            self.halted = True
            if isinstance(error, KeyboardInterrupt):
                raise
        finally:
            try:
                event = self.journal.append(event)
                self.report()
            finally:
                if mode != "conversation":
                    conversation.close()
        return {k: self.journal.clean(v) if isinstance(v, str) else v for k, v in event.items()}

    def comparison_plan(self, *, resume=False):
        rows = [e for e in self.journal.read() if e["kind"] == "turn" and e["mode"] == "comparison"]
        if resume and not rows:
            raise RuntimeError("再開する比較実験がありません。最初は1を選んでください。")
        experiment = rows[-1]["experiment"] if resume else uuid4().hex
        done = {(e["model"], e["question"], e["repeat"]) for e in rows
                if e["experiment"] == experiment and e["status"] == "ok"}
        pending = [(model, question, repeat) for model in MODELS for question in QUESTIONS
                   for repeat in range(1, 4) if (model, question, repeat) not in done]
        return experiment, pending

    def compare(self, on_result=print, *, resume=False):
        if self.halted:
            raise RuntimeError("前のAPIエラーで停止中です。0で終了して開き直してから5を選んでください。")
        experiment, pending = self.comparison_plan(resume=resume)
        if self.max_calls - self.calls < len(pending):
            raise RuntimeError(f"比較実験には残り{len(pending)}回分の枠が必要です。今回は開始しません。")
        for model, question, repeat in pending:
            event = self.call(model, question, mode="comparison", experiment=experiment, repeat=repeat)
            on_result(event)
            if self.halted:
                return experiment
        return experiment
