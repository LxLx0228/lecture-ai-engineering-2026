"""第3回: エージェントに持たせるツール。

ここに書いた関数の docstring と型ヒントが、そのままモデルへの「ツールの説明文」になる。
説明文を書き換えると、モデルがツールを使うかどうか・どんな引数で呼ぶかが変わる。
"""

import json
import os
import random
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from tempfile import NamedTemporaryFile

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LEARNING_LOG_DIR = DATA_DIR.parent / "learning_log"
MAX_RESULTS = 3


@cache
def load_glossary() -> list[dict]:
    with open(DATA_DIR / "glossary.json", encoding="utf-8") as f:
        return json.load(f)["terms"]


def _normalize(text: str) -> str:
    return text.casefold().replace(" ", "").replace("　", "")


def _names(entry: dict) -> list[str]:
    return [_normalize(entry["term"]), _normalize(entry["id"])] + [
        _normalize(a) for a in entry["aliases"]
    ]


def search_glossary(query: str) -> str:
    """講座の用語集から用語を検索し、定義と、どの回で扱うかを返す。

    AIエンジニアリングの用語や、この講座（AIE2026）に固有の言葉について
    答える前に使うこと。query には調べたい用語を1つだけ入れる（例: "LoRA"）。
    """
    q = _normalize(query)
    if not q:
        return "検索語が空です。調べたい用語を1つ指定してください。"
    glossary = load_glossary()

    exact = [e for e in glossary if q in _names(e)]
    # 「r」のような短い別名が何にでも当たらないよう、名前側が短いものは除く
    partial = [
        e for e in glossary
        if e not in exact and any(q in n or (len(n) >= 3 and n in q) for n in _names(e))
    ]
    in_definition = [
        e for e in glossary
        if e not in exact and e not in partial and q in _normalize(e["definition"])
    ]
    hits = (exact + partial + in_definition)[:MAX_RESULTS]

    if not hits:
        return f"用語集に「{query}」は見つかりませんでした。"
    return json.dumps(
        [
            {
                "term": e["term"],
                "lecture": e["lecture"],
                "definition": e["definition"],
                "related": e["related"],
            }
            for e in hits
        ],
        ensure_ascii=False,
    )


def get_lecture_summary(lecture: int) -> str:
    """指定した回（1始まりの整数）の講義の要点を返す。

    「第3回では何を学ぶ？」のように、回の内容や関係を聞かれたときに使う。
    """
    path = DATA_DIR / "lectures" / f"{lecture:02d}.md"
    if not path.exists():
        available = sorted(p.stem for p in (DATA_DIR / "lectures").glob("*.md"))
        return f"第{lecture}回の要点はまだありません。用意されている回: {', '.join(available)}"
    return path.read_text(encoding="utf-8")


def make_quiz(lecture: int) -> str:
    """指定した回の確認問題や小テストを求められたときに使う。

    lecture は1始まりの講義回。用語集から問題と答えを1組返す。
    出題時は問題だけを表示し、答えは学習者が回答するまで明かさない。
    """
    terms = [entry for entry in load_glossary() if entry["lecture"] == lecture]
    if not terms:
        return f"第{lecture}回の用語が見つからないため、確認問題を作れません。"
    entry = random.choice(terms)
    return json.dumps(
        {"question": f"「{entry['term']}」の意味は？", "answer": entry["definition"]},
        ensure_ascii=False,
    )


def save_learning_log(summary: str) -> str:
    """利用者が学習記録の保存を求めたとき、学んだことの要約を保存する。

    summary は保存する記録の本文。既存の記録を上書きせず、1件ずつ保存する。
    """
    summary = summary.strip()
    if not summary:
        return "学習記録の本文が空のため、保存できません。"

    LEARNING_LOG_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    file = NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=LEARNING_LOG_DIR,
        prefix=now.strftime("%Y%m%dT%H%M%SZ-"),
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(file.name)
    log_path = temporary_path.with_suffix(".md")
    try:
        with file:
            file.write(f"# 学習記録\n\n{summary}\n")
        os.link(temporary_path, log_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return f"学習記録を learning_log/{log_path.name} に保存しました。"
