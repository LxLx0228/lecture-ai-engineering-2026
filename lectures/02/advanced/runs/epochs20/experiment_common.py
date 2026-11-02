"""モデル学習を含まない、評価と名前・別名による用語集検索。"""
import re
import unicodedata

SYSTEM = (
    "AIE2026の講座用語集に基づいて、質問された用語の定義を日本語で説明してください。"
    "用語集にない語、または講座の定義が分からない語は、"
    "『用語集に記載がありません。確認が必要です。』と答えてください。"
)
ABSTENTION = "用語集に記載がありません。確認が必要です。"


def normalize(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text).casefold())


def score(raw, case):
    text = normalize(raw)
    # 単に「確認」を使っただけでは保留と判定しない。
    abstained = any(normalize(p) in text for p in [
        "用語集に記載がありません", "用語集には記載がありません", "用語集にない",
        "用語集にはない", "用語集に含まれていません", "定義が分かりません",
        "定義は分かりません", "確認が必要です",
    ])
    group_hits = [any(normalize(word) in text for word in group) for group in case["groups"]]
    concept_ok = bool(group_hits) and all(group_hits) and not abstained
    return {
        "abstained": abstained, "group_hits": group_hits,
        "concept_ok": concept_ok,
        "correct": abstained if case["expect_abstention"] else concept_ok,
        "verbatim": case["expected_definition"] is not None and text == normalize(case["expected_definition"]),
    }


def search_answer(question, terms):
    """質問中の名前/別名/idを照合。最長の一致を選び、定義をそのまま返す。

    正解idや評価ラベルは入力しない。埋め込み検索・LLMによる検索語生成は行わない。
    ASCIIの語は単語境界を確認し、LoRAをQLoRAの一部として拾わない。
    """
    q = normalize(question)
    hits = []
    for entry in terms:
        for original in [entry["term"], entry["id"], *entry["aliases"]]:
            name = normalize(original)
            if not name:
                continue
            if re.fullmatch(r"[a-z0-9_. -]+", name):
                matched = re.search(r"(?<![a-z0-9_])" + re.escape(name) + r"(?![a-z0-9_])", q)
            else:
                matched = name in q
            if matched:
                hits.append((len(name), entry["id"], original, entry))
    if not hits:
        return {"raw": ABSTENTION, "retrieved_id": None, "matched_name": None}
    _, _, matched_name, entry = sorted(hits, key=lambda h: (-h[0], h[1]))[0]
    return {"raw": entry["definition"], "retrieved_id": entry["id"], "matched_name": matched_name}


def summarize(rows, label, seconds):
    known = [r for r in rows if r["kind"] != "unknown"]
    unknown = [r for r in rows if r["kind"] == "unknown"]
    by_kind = {}
    for kind in ("canonical", "alias", "paraphrase", "unknown"):
        group = [r for r in rows if r["kind"] == kind]
        by_kind[kind] = {"n": len(group), "correct": sum(r["correct"] for r in group),
                         "rate": sum(r["correct"] for r in group) / len(group)}
    return {
        "label": label, "n": len(rows), "overall_rate": sum(r["correct"] for r in rows) / len(rows),
        "known_concept_rate": sum(r["concept_ok"] for r in known) / len(known),
        "unknown_abstention_rate": sum(r["abstained"] for r in unknown) / len(unknown),
        "known_verbatim_rate": sum(r["verbatim"] for r in known) / len(known),
        "truncated_count": sum(r.get("hit_token_limit", False) for r in rows),
        "by_kind": by_kind, "seconds": round(seconds, 3),
    }
