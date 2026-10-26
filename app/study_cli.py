"""初心者向けの第1回学習メニュー。キーはメモリ上だけで扱う。"""

import argparse
from datetime import datetime
import fcntl
import getpass
import os
from pathlib import Path
import sys
import warnings
from zoneinfo import ZoneInfo

from app.study import StudySession, ROOT, MODELS, FREE_TIER, cell

MEMORY_QUESTIONS = (
    "これから会話の確認をします。私の好きな動物はカワウソです。この会話の間だけ覚えておいてください。",
    "さっき伝えた、私の好きな動物は何でしたか？",
)


def display(session, event):
    print(f"\n[{session.calls}/30] {event['model']} / {event['mode']}")
    print("質問: " + session.journal.clean(event["question"]))
    print(session.journal.clean(event.get("answer") or "（回答本文なし）"))
    if event.get("elapsed_seconds") is not None:
        print(f"応答 {event['elapsed_seconds']:.3f}秒 / 入力 {cell(event.get('input_tokens'))} / "
              f"出力 {cell(event.get('output_tokens'))} / 思考 {cell(event.get('thinking_tokens'))} / 合計 {cell(event.get('total_tokens'))}")
    if event.get("status") != "ok":
        print("状態: " + event["status"] + " " + session.journal.clean(event.get("error", "")))
        if event.get("error_code") == 503:
            print("503: モデル側の一時的な混雑・サービス利用不可です。成功済みの結果は保存されています。")
        elif event.get("error_code") == 429:
            print("429: 無料枠のレート制限です。AI Studioで利用枠を確認してください。")
        elif event.get("error_code") == 504:
            print("504: API処理が時間内に完了しませんでした。キーが無効というエラーではありません。")
        elif event.get("error_code") == 401:
            print("401: キーの認証が拒否されました。AI Studioのコピーボタンでキー全文をコピーし直してください。")
    if event.get("status") == "truncated":
        print("出力上限や保存上限による省略の可能性があります。レポートにも記録しました。")
    print("保存しました。")


def run_all(session, *, resume=False):
    experiment, pending = session.comparison_plan(resume=resume)
    memory_done = resume and any(
        e["kind"] == "turn" and e["mode"] == "conversation" and e["status"] == "ok"
        and e.get("experiment") == experiment and e["question"] == MEMORY_QUESTIONS[1]
        and e.get("history_messages", 0) >= 2 for e in session.journal.read()
    )
    needed = len(pending) + (0 if memory_done else 2)
    if needed == 0:
        print("直近の実験は全件保存済みです。追加のAPI呼び出しはしません。3で気づきを追記できます。")
        return
    if session.max_calls - session.calls < needed:
        raise RuntimeError(f"残りの実験には{needed}回分の枠が必要です。今回は開始しません。")
    print(f"比較は残り{len(pending)}回、会話確認は{0 if memory_done else 2}回です。開始間隔15秒。")
    print("今回使う比較モデル: " + " / ".join(dict.fromkeys(model for model, _, _ in pending)))
    experiment = session.compare(lambda event: display(session, event), resume=resume)
    if session.halted:
        print("APIエラーで中断しました。ここまでの結果は保存済み。0で終了し、後で開き直して5で続きを実行できます。自動再試行はしません。")
        return
    if memory_done:
        return
    session.reset()
    print("続いて同じモデルへの2往復で会話履歴を確認します。")
    for question in MEMORY_QUESTIONS:
        display(session, session.call(MODELS[0], question, experiment=experiment))
        if session.halted:
            return
    session.reset()
    print("実験記録と下書きを保存しました。回答が正しいかを確認し、メニュー3で気づきを追記してください。")


def chat(session):
    print("続けて質問できます。/newで履歴を消去、/menuでメニューへ、/modelでモデルを切替。")
    model = MODELS[0]
    while not session.halted:
        question = input("\n質問> ").strip()
        if question == "/menu" or question == "/quit":
            return
        if question == "/new":
            session.reset()
            print("会話履歴を消しました。保存した記録は残ります。")
            continue
        if question == "/model":
            model = MODELS[1] if model == MODELS[0] else MODELS[0]
            session.reset()
            print("新しい会話のモデル: " + model)
            continue
        if question:
            try:
                display(session, session.call(model, question))
            except ValueError as error:
                print(error)


def menu(session):
    session.report()
    while True:
        print("\n1: 基礎・実装・発展をまとめて実験（最大14回）")
        print("2: 自由に質問する（会話履歴と結果を保存）")
        print("3: 自分の気づき・難しかった点を追記（API呼び出しなし）")
        print("4: 保存先を表示・下書きを更新（API呼び出しなし）")
        print("5: 中断した実験の続き（成功済みの質問は飛ばす）")
        print("0: 保存して終了")
        choice = input("番号> ").strip()
        try:
            if choice == "0":
                return
            if choice == "1":
                run_all(session)
            elif choice == "2":
                chat(session)
            elif choice == "3":
                session.note(input("気づき（最大2000文字）> "))
                print("追記しました。")
            elif choice == "4":
                print("講座内の下書き: " + str(session.report()))
                if session.export_path:
                    print("取り出し用コピー: " + str(session.export_path))
                print("API呼び出し回数: " + str(session.calls) + "/30")
            elif choice == "5":
                run_all(session, resume=True)
            else:
                print("0〜5の番号を入力してください。")
        except (ValueError, RuntimeError) as error:
            print(session.journal.clean(error))


def main(argv=None):
    parser = argparse.ArgumentParser(description="AIE第1回の質問・比較・自動記録")
    parser.add_argument("--export", type=Path, help="提出用answer.mdのコピー先")
    parser.add_argument("--check", action="store_true", help="APIを呼ばず起動確認")
    args = parser.parse_args(argv)
    if args.check:
        print("起動確認OK。2モデル、12回比較、会話履歴、レポート機能を読み込みました。API呼び出しなし。")
        return
    if not sys.stdin.isatty():
        raise SystemExit("ターミナルから起動してください。キーを引数やパイプで渡さないでください。")
    lock_path = ROOT / ".logs/lec01/active.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("別の学習メニューが起動中です。先にそちらを0で終了してください。")
        print("AIE 第1回 自動記録メニュー")
        print("無料枠のモデル2つだけを使用。1起動30回まで、1回答2048トークンまで。エラーの自動再試行なし。")
        print("キーは入力中も非表示。ファイルに保存せず、この起動中だけ使用します。")
        today = datetime.now(ZoneInfo("Asia/Tokyo")).date().isoformat()
        if today != FREE_TIER["checked_on"]:
            print("モデルの無料枠の確認日は " + FREE_TIER["checked_on"] + " です。")
            print("https://aistudio.google.com/rate-limit で今日の無料枠と2モデルの利用枠を確認してください。")
            if input("確認できた場合だけ「無料枠確認」と入力> ").strip() != "無料枠確認":
                return
        print("課金設定をしていない無料枠プロジェクトのAPIキーを使ってください。")
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            key = getpass.getpass("Gemini APIキーを貼り付けてEnter（文字は表示されません）: ").strip()
        if not key:
            return
        session = StudySession(key, export_path=args.export)
        try:
            menu(session)
        except (KeyboardInterrupt, EOFError):
            print("\n中断しました。保存済みの結果は残ります。")
        finally:
            session.reset()
            session.report()
            print("保存先: " + str(args.export or ROOT / "lectures/01/answer.md"))


if __name__ == "__main__":
    main()
