"""第1回: 単発質問と会話履歴。学習記録は python -m app.study_cli。"""

import argparse
from dataclasses import dataclass
from time import perf_counter
from google import genai
from google.genai import types, errors
from app.config import MODEL, api_error_message, get_api_key

DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"
MAX_QUESTION_CHARS = 2000
MAX_CONTEXT_CHARS = 12000
MAX_OUTPUT_TOKENS = 2048


@dataclass
class Reply:
    text: str
    elapsed_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    thinking_tokens: int | None
    total_tokens: int | None
    history_messages: int
    finish_reason: str


class Conversation:
    """成功した会話だけ保持。比較実験には毎回別インスタンスを使う。"""

    def __init__(self, model=MODEL, api_key=None, *, client=None):
        self.model = model
        self.history = []
        self.client = client or genai.Client(
            vertexai=False,
            api_key=api_key or get_api_key(),
            http_options=types.HttpOptions(
                timeout=60000, retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )

    def reset(self):
        self.history.clear()

    def close(self):
        self.client.close()

    def validate(self, question):
        if not question.strip() or len(question) > MAX_QUESTION_CHARS:
            raise ValueError("質問は1〜2000文字にしてください。")
        context_size = sum(len(p.text or "") for c in self.history for p in c.parts)
        if context_size + len(question) > MAX_CONTEXT_CHARS:
            raise ValueError("会話の長さが上限です。/new で新しい会話にしてください。")

    def send(self, question):
        self.validate(question)
        user = types.Content(role="user", parts=[types.Part.from_text(text=question)])
        history_messages = len(self.history)
        start = perf_counter()
        response = self.client.models.generate_content(
            model=self.model, contents=[*self.history, user],
            config=types.GenerateContentConfig(
                max_output_tokens=MAX_OUTPUT_TOKENS,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        elapsed = perf_counter() - start
        text = response.text or ""
        usage = response.usage_metadata
        candidate = (response.candidates or [None])[0]
        reason = getattr(candidate, "finish_reason", None)
        if text:
            self.history.extend([
                user, types.Content(role="model", parts=[types.Part.from_text(text=text)])
            ])
        return Reply(
            text=text, elapsed_seconds=elapsed,
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            thinking_tokens=getattr(usage, "thoughts_token_count", None),
            total_tokens=getattr(usage, "total_token_count", None),
            history_messages=history_messages,
            finish_reason=getattr(reason, "value", str(reason or "不明")),
        )


def metrics(reply):
    def value(v):
        return "未取得" if v is None else str(v)
    return (
        f"応答時間 {reply.elapsed_seconds:.3f}秒 / 入力 {value(reply.input_tokens)} / "
        f"出力 {value(reply.output_tokens)} / 思考 {value(reply.thinking_tokens)} / "
        f"合計 {value(reply.total_tokens)} トークン"
    )


def ask(question):
    conversation = Conversation()
    try:
        reply = conversation.send(question)
        print(metrics(reply))
        return reply.text
    finally:
        conversation.close()


def main():
    parser = argparse.ArgumentParser(description="LLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--chat", action="store_true", help="履歴を保持して会話する（最大30回）")
    args = parser.parse_args()
    conversation = Conversation()
    try:
        for _ in range(30 if args.chat else 1):
            question = input("質問（/quitで終了、/newで履歴を消去）> ") if args.chat else args.question
            if question == "/quit":
                break
            if question == "/new":
                conversation.reset()
                continue
            try:
                reply = conversation.send(question)
                print(reply.text or "（回答本文なし）")
                print(metrics(reply))
            except errors.APIError as error:
                print(api_error_message(error.code, "詳細はAPI側で確認してください。"))
                break
            except ValueError as error:
                print(error)
    except (KeyboardInterrupt, EOFError):
        print("\n終了しました。")
    finally:
        conversation.close()


if __name__ == "__main__":
    main()
