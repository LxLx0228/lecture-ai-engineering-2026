"""保存した生成原文から講義の指標を再計算する（推論・学習は行わない）。"""
import json
import re
from pathlib import Path


def verify(root: Path):
    load = lambda name: json.loads((root / name).read_text(encoding="utf-8"))
    splits = {
        name: [json.loads(line) for line in (root / f"{name}.jsonl").read_text().splitlines()]
        for name in ("train", "valid", "test")
    }
    assert {k: len(v) for k, v in splits.items()} == {"train": 14, "valid": 7, "test": 21}
    questions = [row["question"] for rows in splits.values() for row in rows]
    assert len(questions) == len(set(questions)), "split間に同一質問がある"
    gold, result = load("gold.json"), load("result.json")
    assert set(gold) == {r["question"] for r in splits["test"]}
    assert result["executed_training_myself"] is True
    assert result["config"]["revision"] == "2fc06364715b967f1860aea9cf38778875588b17"
    evaluations = {}
    for label in ("before", "after"):
        ev = load(f"eval_{label}.json")
        assert [r["question"] for r in ev["rows"]] == [r["question"] for r in splits["test"]]
        for row in ev["rows"]:
            try:
                obj = json.loads(row["raw"])
            except (ValueError, TypeError):
                obj = None
            fmt = (isinstance(obj, dict) and set(obj) == {"answer", "needs_review"}
                   and isinstance(obj["answer"], str) and type(obj["needs_review"]) is bool)
            loose = obj
            if not fmt:
                match = re.search(r"\{.*\}", row["raw"], re.S)
                try:
                    loose = json.loads(match.group()) if match else None
                except ValueError:
                    loose = None
            expected = gold[row["question"]]
            content = (isinstance(loose, dict) and loose.get("needs_review") == expected["needs_review"]
                       and all(word in loose.get("answer", "") for word in expected["must_include"]))
            assert row["format_ok"] == fmt, row["question"]
            assert row["content_ok"] == content, row["question"]
            assert row["both_ok"] == (fmt and content)
            assert row["gold_needs_review"] == expected["needs_review"]
        assert ev["summary"]["n"] == 21
        for metric, flag in (("format_rate", "format_ok"), ("content_rate", "content_ok"), ("both_rate", "both_ok")):
            calculated = sum(r[flag] for r in ev["rows"]) / 21
            assert abs(calculated - ev["summary"][metric]) < 1e-12
        assert result[label] == ev["summary"]
        evaluations[label] = ev
        print(label, json.dumps(ev["summary"], ensure_ascii=False))
    regressions = [b["question"] for b, a in zip(evaluations["before"]["rows"], evaluations["after"]["rows"])
                   if b["both_ok"] and not a["both_ok"]]
    assert result["regressed"] == regressions
    assert len(result["knowledge"]) == 4
    assert result["training_metrics"]["epoch"] == 20
    print("PASS: 質問分離、原文からの全21問再採点、前後要約、退行、20エポック完了を確認")


if __name__ == "__main__":
    verify(Path(__file__).parent / "results")
