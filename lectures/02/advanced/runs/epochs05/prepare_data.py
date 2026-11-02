"""講座スクリプトをそのまま実行し、質問表現でtrain/validを分ける。"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from evaluation_cases import make_cases
from experiment_common import SYSTEM, score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def main():
    data = HERE / "data"
    data.mkdir(exist_ok=True)
    glossary_bytes = (REPO / "data/glossary.json").read_bytes()
    (data / "glossary_snapshot.json").write_bytes(glossary_bytes)
    script = REPO / "scripts/make_sft_dataset.py"
    raw = subprocess.run([sys.executable, str(script)], check=True, capture_output=True).stdout
    (data / "glossary_sft.jsonl").write_bytes(raw)
    records = [json.loads(line) for line in raw.decode().splitlines()]
    terms = json.loads(glossary_bytes)["terms"]
    assert len(records) == 2 * len(terms) == 116
    train = [{"messages": [{"role": "system", "content": SYSTEM}] + r["messages"]} for r in records[::2]]
    valid = [{"messages": [{"role": "system", "content": SYSTEM}] + r["messages"]} for r in records[1::2]]
    cases = make_cases(terms)
    questions = [r["messages"][-2]["content"] for r in train + valid] + [c["question"] for c in cases]
    assert len(questions) == len(set(questions))
    assert len(cases) == 42
    for c in cases:
        if not c["expect_abstention"]:
            assert score(c["expected_definition"], c)["correct"], c["id"]
    for name, rows in [("train.jsonl", train), ("valid.jsonl", valid)]:
        (data / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (data / "eval_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "glossary_version": json.loads(glossary_bytes)["version"], "terms": len(terms),
        "generator": "scripts/make_sft_dataset.py (unmodified)",
        "generator_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "generated_rows": len(records), "train": len(train), "valid": len(valid), "test": len(cases),
        "split": "first template per term for train; second template for valid; all 58 terms shared",
        "evaluation": "12 canonical + 12 unseen alias questions + 12 descriptive paraphrases + 6 out-of-glossary questions",
        "training_unknown_examples": 0,
        "input_knowledge": "No glossary definitions, known-term index, retrieved text, expected answers or gold concepts in inference prompts.",
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data.iterdir()) if p.is_file() and p.name != "manifest.json"},
    }
    (data / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "sha256"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
