"""2回の実験の保存出力と入力スナップショットを再検証する。GPU不要。"""
import hashlib
import json
import math
from pathlib import Path
from experiment_common import score, search_answer, summarize

HERE = Path(__file__).resolve().parent

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def verify_run(root, epochs, steps, audit=False):
    terms = load(root / "data/glossary_snapshot.json")["terms"]
    cases = load(root / "data/eval_cases.json")
    result = load(root / "results/result.json")
    manifest = load(root / "results/run_manifest.json")
    # 後から改良した検証コードと、実行時に存在したソースは区別する。
    for name, expected in manifest.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, (root, name)
    questions = []
    for name in ("train", "valid"):
        rows = [json.loads(line) for line in (root / f"data/{name}.jsonl").read_text().splitlines()]
        assert len(rows) == 58
        assert {r["messages"][-1]["content"] for r in rows} == {t["definition"] for t in terms}
        questions.extend(r["messages"][-2]["content"] for r in rows)
    assert len(cases) == 42
    questions.extend(c["question"] for c in cases)
    groups = [("", cases)]
    if audit:
        audit_cases = load(root / "data/eval_cases_final.json")
        assert len(audit_cases) == 12
        questions.extend(c["question"] for c in audit_cases)
        groups.append(("audit_", audit_cases))
    assert len(questions) == len(set(questions))
    for prefix, case_group in groups:
        evaluations = {}
        for method in ("before", "after", "search"):
            label = prefix + method
            value = load(root / f"results/eval_{label}.json")
            assert [r["id"] for r in value["rows"]] == [c["id"] for c in case_group]
            for row, case in zip(value["rows"], case_group):
                assert (row["question"], row["kind"]) == (case["question"], case["kind"])
                for name, expected in score(row["raw"], case).items():
                    assert row[name] == expected, (root, label, row["id"], name)
                if method == "search":
                    for name, expected in search_answer(case["question"], terms).items():
                        assert row[name] == expected
                else:
                    assert 0 < row["generated_tokens"] <= 256
            assert summarize(value["rows"], label, value["summary"]["seconds"]) == value["summary"]
            assert result[label] == value["summary"]
            evaluations[method] = value
            print(root.name, label, value["summary"]["overall_rate"])
        if not prefix:
            expected = [b["id"] for b, a in zip(evaluations["before"]["rows"], evaluations["after"]["rows"])
                        if b["correct"] and not a["correct"]]
            assert result["regressed_ids"] == expected
    assert result["executed_training_myself"] is True
    assert result["global_step"] == steps
    assert result["training_metrics"]["epoch"] == epochs
    assert all(math.isfinite(r["loss"]) for r in result["trainer_log_history"] if "loss" in r)
    assert result["config"]["revision"] == "2fc06364715b967f1860aea9cf38778875588b17"
    # 原本の長さ記録はlen(BatchEncoding)でキー数を数えていた。事後にinput_idsで再計測した別記録を使う。
    lengths = load(root / "results/token_lengths_verified.json")
    assert len(lengths["train"]) == len(lengths["valid"]) == 58
    assert max(lengths["train"] + lengths["valid"]) == lengths["max"] <= 512
    for split in ("train", "valid"):
        assert hashlib.sha256((root / f"data/{split}.jsonl").read_bytes()).hexdigest() == lengths[f"{split}_sha256"]
    observed_token_totals = [r["num_tokens"] for r in result["trainer_log_history"] if "num_tokens" in r]
    assert observed_token_totals[-1] == sum(lengths["train"]) * epochs
    assert all(r["matches_before"] for r in result["adapter_disabled_checks"])
    if audit:
        assert all(r["matches_before"] for r in result["baseline_rechecks"])
        losses = [r["eval_loss"] for r in result["trainer_log_history"] if "eval_loss" in r]
        assert len(losses) == 4
        assert result["best_validation_loss"] == min(losses)
        assert load(root / "data/baseline_before.json") == load(root / "results/eval_before.json")
    return result

def main():
    first, final = HERE / "runs/epochs05", HERE / "runs/epochs20"
    verify_run(first, 5, 40)
    verify_run(final, 20, 160, audit=True)
    for name in ("train.jsonl", "valid.jsonl", "eval_cases.json", "glossary_snapshot.json"):
        assert (first / "data" / name).read_bytes() == (final / "data" / name).read_bytes()
    for p in (final / "results").glob("*.json"):
        assert p.read_bytes() == (HERE / "results" / p.name).read_bytes()
    print("PASS: 両実験の入力SHA256・質問分離・全出力再採点・検索再現・学習完了・検証損失による選択・基盤出力一致。元42問と訓練データは変更なし。")

if __name__ == "__main__":
    main()
