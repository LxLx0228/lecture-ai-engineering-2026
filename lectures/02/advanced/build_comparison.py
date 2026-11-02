"""生成原文を読みやすいMarkdownに並べる。再推論・再採点・学習は行わない。"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    assert load(HERE / "results/result.json")["config"]["num_train_epochs"] == 20
    sections = ["# 発展課題：生成原文の比較\n", "語群一致は意味の正確さの代理指標。初回42問と追加12問を分けて読む。\n"]
    for prefix, case_name in [("", "eval_cases.json"), ("audit_", "eval_cases_final.json")]:
        cases = load(HERE / "data" / case_name)
        methods = [
            ("基盤モデル", load(HERE / f"results/eval_{prefix}before.json")["rows"]),
            ("20エポック後", load(HERE / f"results/eval_{prefix}after.json")["rows"]),
            ("名前・別名検索", load(HERE / f"results/eval_{prefix}search.json")["rows"]),
        ]
        if not prefix:
            methods.insert(1, ("初回5エポック後", load(HERE / "runs/epochs05/results/eval_after.json")["rows"]))
        indexed = [(name, {r["id"]: r for r in rows}) for name, rows in methods]
        sections.append("\n## " + ("追試前に固定した追加12問" if prefix else "初回から固定した42問（追試は探索的）") + "\n")
        for case in cases:
            sections.append(f"\n### {case['id']}\n\n質問：{case['question']}\n\n")
            if case["expect_abstention"]:
                sections.append("期待：講座用語集の未掲載を認めて保留する。\n")
            else:
                sections.append("用語集の定義：" + case["expected_definition"] + "\n")
            for name, rows in indexed:
                row = rows[case["id"]]
                flags = f"条件一致={row['correct']}、保留検出={row['abstained']}"
                if "hit_token_limit" in row:
                    flags += f"、上限による途中切れ={row['hit_token_limit']}"
                sections.append(f"\n**{name}**（{flags}）\n\n~~~~text\n{row['raw']}\n~~~~\n")
    (HERE / "comparison.md").write_text("".join(sections), encoding="utf-8")
    print("comparison.md created from saved raw outputs")


if __name__ == "__main__":
    main()
