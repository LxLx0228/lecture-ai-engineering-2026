"""保存した生成原文・検索結果・入力ハッシュ・集計の整合性を検証する。GPU不要。"""
import hashlib
import json
import math
from pathlib import Path

from experiment_common import score, search_answer, summarize

HERE = Path(__file__).resolve().parent


def main():
    load = lambda p: json.loads(p.read_text(encoding="utf-8"))
    terms = load(HERE / "data/glossary_snapshot.json")["terms"]
    cases = load(HERE / "data/eval_cases.json")
    result = load(HERE / "results/result.json")
    manifest = load(HERE / "results/run_manifest.json")
    for name, expected in manifest.items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected, name
    for name in ("train", "valid"):
        records = [json.loads(line) for line in (HERE / f"data/{name}.jsonl").read_text().splitlines()]
        assert len(records) == 58
        assert {r["messages"][-1]["content"] for r in records} == {t["definition"] for t in terms}
    train_questions = [json.loads(line)["messages"][-2]["content"]
                       for name in ("train", "valid") for line in (HERE / f"data/{name}.jsonl").read_text().splitlines()]
    all_questions = train_questions + [c["question"] for c in cases]
    assert len(all_questions) == len(set(all_questions))
    assert len(cases) == 42
    evaluations = {}
    for label in ("before", "after", "search"):
        value = load(HERE / f"results/eval_{label}.json")
        assert [r["id"] for r in value["rows"]] == [c["id"] for c in cases]
        for row, case in zip(value["rows"], cases):
            assert row["question"] == case["question"]
            assert row["kind"] == case["kind"]
            for name, expected in score(row["raw"], case).items():
                assert row[name] == expected, (label, row["id"], name)
            if label == "search":
                for name, expected in search_answer(case["question"], terms).items():
                    assert row[name] == expected
            else:
                assert 0 < row["generated_tokens"] <= 256
        assert summarize(value["rows"], label, value["summary"]["seconds"]) == value["summary"]
        assert result[label] == value["summary"]
        evaluations[label] = value
        print(label, json.dumps(value["summary"], ensure_ascii=False))
    expected_regressions = [b["id"] for b, a in zip(evaluations["before"]["rows"], evaluations["after"]["rows"])
                            if b["correct"] and not a["correct"]]
    assert result["regressed_ids"] == expected_regressions
    assert result["executed_training_myself"] is True
    assert result["global_step"] == 40
    assert result["training_metrics"]["epoch"] == 5
    assert all(math.isfinite(row["loss"]) for row in result["trainer_log_history"] if "loss" in row)
    assert result["config"]["revision"] == "2fc06364715b967f1860aea9cf38778875588b17"
    lengths = load(HERE / "results/token_lengths.json")
    assert len(lengths["train"]) == len(lengths["valid"]) == 58
    assert max(lengths["train"] + lengths["valid"]) == lengths["max"] <= 512
    assert all(row["matches_before"] for row in result["adapter_disabled_checks"])
    print("PASS: 入力ソースのSHA256、質問分離、126出力の再採点・集計、検索再現、学習40ステップ、長さ上限、adapter無効時の基盤出力一致")


if __name__ == "__main__":
    main()
