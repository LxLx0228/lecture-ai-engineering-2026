"""Colab T4で1回だけ学習し、閉じた本での定義回答と検索を比較する。"""
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset, disable_progress_bars
from huggingface_hub.utils import disable_progress_bars as disable_hub_progress_bars
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.utils import logging as transformers_logging
from trl import SFTConfig, SFTTrainer

from experiment_common import SYSTEM, score, search_answer, summarize

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results20"
CONFIG = {
    "base_model": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "seed": 42, "num_train_epochs": 20, "warmup_steps": 6,
    "lr": 2e-4, "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    "batch_size": 4, "gradient_accumulation_steps": 2,
    "max_seq_len": 512, "max_new_tokens": 256,
    "do_sample": False, "enable_thinking": False, "dtype": "float32",
    "eval_every_steps": 40, "select_checkpoint": "lowest valid loss",
}


def save(name, value):
    (RESULTS / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(name):
    return [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines()]


def generate(model, tokenizer, question):
    # 正解・定義・検索結果はこのmessagesに含めない。
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=CONFIG["max_new_tokens"], do_sample=False,
                                pad_token_id=tokenizer.pad_token_id)
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    eos = model.generation_config.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    ended = bool(len(new_tokens) and int(new_tokens[-1]) in eos_ids)
    return {
        "raw": tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
        "input_tokens": inputs["input_ids"].shape[1], "generated_tokens": len(new_tokens),
        "hit_token_limit": len(new_tokens) >= CONFIG["max_new_tokens"] and not ended,
    }


def evaluate(model, tokenizer, cases, label):
    rows = []
    started = time.perf_counter()
    for i, case in enumerate(cases):
        generated = generate(model, tokenizer, case["question"])
        rows.append({"id": case["id"], "kind": case["kind"], "question": case["question"],
                     **generated, **score(generated["raw"], case)})
        if (i + 1) % 6 == 0:
            print(f"{label}: {i + 1}/{len(cases)} questions done", flush=True)
    summary = summarize(rows, label, time.perf_counter() - started)
    value = {"summary": summary, "rows": rows}
    save(f"eval_{label}.json", value)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return value


def main():
    # Colabの出力ウィジェット読込エラーを避け、進捗を標準出力だけに残す。
    disable_progress_bars()
    disable_hub_progress_bars()
    transformers_logging.disable_progress_bar()
    assert torch.cuda.is_available(), "T4 GPUを選択してください。CPUでは学習しません。"
    assert not (RESULTS / "result.json").exists(), "完了済みの結果があります。再実行する場合は別ランタイムで行ってください。"
    RESULTS.mkdir(exist_ok=True)
    set_seed(CONFIG["seed"])
    terms = json.loads((DATA / "glossary_snapshot.json").read_text())["terms"]
    cases = json.loads((DATA / "eval_cases.json").read_text())
    audit_cases = json.loads((DATA / "eval_cases_final.json").read_text())
    train, valid = read_jsonl("train.jsonl"), read_jsonl("valid.jsonl")
    assert (len(terms), len(train), len(valid), len(cases)) == (58, 58, 58, 42)
    # 検索はモデルなし。正解idを渡さず、同じ質問文だけを渡す。
    t0 = time.perf_counter()
    search_rows = []
    for case in cases:
        found = search_answer(case["question"], terms)
        search_rows.append({"id": case["id"], "kind": case["kind"], "question": case["question"],
                            **found, **score(found["raw"], case)})
    search = {"summary": summarize(search_rows, "search", time.perf_counter() - t0), "rows": search_rows}
    save("eval_search.json", search)
    t0 = time.perf_counter()
    audit_search_rows = []
    for case in audit_cases:
        found = search_answer(case["question"], terms)
        audit_search_rows.append({"id": case["id"], "kind": case["kind"], "question": case["question"],
                                  **found, **score(found["raw"], case)})
    audit_search = {"summary": summarize(audit_search_rows, "audit_search", time.perf_counter() - t0), "rows": audit_search_rows}
    save("eval_audit_search.json", audit_search)
    print("data:", len(train), len(valid), len(cases), flush=True)
    print("CONFIG:", json.dumps(CONFIG), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"], revision=CONFIG["revision"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lengths = [len(tokenizer.apply_chat_template(row["messages"], tokenize=True)) for row in train + valid]
    # 長い正解が切り落とされていないことを学習開始前に確認する。
    assert max(lengths) <= CONFIG["max_seq_len"], f"max tokens {max(lengths)} > max_seq_len"
    save("token_lengths.json", {"train": lengths[:58], "valid": lengths[58:], "max": max(lengths)})
    model = AutoModelForCausalLM.from_pretrained(CONFIG["base_model"], revision=CONFIG["revision"], dtype=torch.float32).to("cuda")
    model.config.use_cache = True
    model.eval()
    # Trainerがtokenizerに合わせてEOSを変更する前後で生成条件が変わらないよう、ここで先に合わせる。
    model.config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    # 基盤モデルは同一revision・同一生成条件なので初回の42問を再利用する。
    # 2問を再生成して未学習の状態で一致を確認し、残りを重複実行しない。
    before = json.loads((DATA / "baseline_before.json").read_text())
    baseline_checks = []
    for idx in [0, 33]:
        raw = generate(model, tokenizer, cases[idx]["question"])["raw"]
        baseline_checks.append({"id": cases[idx]["id"], "matches_before": raw == before["rows"][idx]["raw"], "raw": raw})
    assert all(row["matches_before"] for row in baseline_checks)
    save("eval_before.json", before)
    audit_before = evaluate(model, tokenizer, audit_cases, "audit_before")
    model.config.use_cache = False
    lora = LoraConfig(r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"], lora_dropout=CONFIG["lora_dropout"],
                      bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    args = SFTConfig(output_dir=str(HERE / "checkpoints"), num_train_epochs=CONFIG["num_train_epochs"],
                     per_device_train_batch_size=4, gradient_accumulation_steps=2, learning_rate=CONFIG["lr"],
                     lr_scheduler_type="cosine", warmup_steps=6, logging_steps=10,
                     eval_strategy="steps", eval_steps=40, save_strategy="steps", save_steps=40,
                     save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
                     fp16=False, max_length=512, packing=False, report_to="none", seed=42, disable_tqdm=True)
    trainer = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list(train),
                         eval_dataset=Dataset.from_list(valid), peft_config=lora, processing_class=tokenizer)
    trainable, total = trainer.model.get_nb_trainable_parameters()
    print(f"trainable params: {trainable} / {total}", flush=True)
    t0 = time.perf_counter()
    train_output = trainer.train()
    training_seconds = time.perf_counter() - t0
    expected_steps = math.ceil(len(train) / 8) * CONFIG["num_train_epochs"]
    assert trainer.state.global_step == expected_steps == 160
    trainer.model.save_pretrained(HERE / "adapter")
    tuned = trainer.model
    tuned.config.use_cache = True
    tuned.eval()
    after = evaluate(tuned, tokenizer, cases, "after")
    audit_after = evaluate(tuned, tokenizer, audit_cases, "audit_after")
    # 同じモデルインスタンスでadapterを無効にした出力を2問だけ取り直し、比較の汚染を検査する。
    adapter_checks = []
    with tuned.disable_adapter():
        for idx in [0, 33]:
            raw = generate(tuned, tokenizer, cases[idx]["question"])["raw"]
            adapter_checks.append({"id": cases[idx]["id"], "matches_before": raw == before["rows"][idx]["raw"], "raw": raw})
    result = {
        "task_level": "Advanced", "executed_training_myself": True,
        "config": CONFIG, "gpu": torch.cuda.get_device_name(0),
        "versions": {name: importlib.metadata.version(name) for name in ["transformers", "peft", "trl", "accelerate", "datasets", "torch"]},
        "data_counts": {"glossary_terms": 58, "raw_generated": 116, "train": 58, "valid": 58, "test": 42, "final_audit": 12},
        "training_seconds": training_seconds, "training_metrics": train_output.metrics,
        "trainer_log_history": trainer.state.log_history, "global_step": trainer.state.global_step,
        "trainable_params": trainable, "total_params": total,
        "before": before["summary"], "after": after["summary"], "search": search["summary"],
        "audit_before": audit_before["summary"], "audit_after": audit_after["summary"], "audit_search": audit_search["summary"],
        "baseline_source": "Initial 5-epoch experiment's base-model outputs, copied unchanged. Two outputs rechecked before training.",
        "baseline_rechecks": baseline_checks,
        "best_validation_loss": trainer.state.best_metric,
        "selected_checkpoint": Path(trainer.state.best_model_checkpoint).name,
        "regressed_ids": [b["id"] for b, a in zip(before["rows"], after["rows"]) if b["correct"] and not a["correct"]],
        "adapter_disabled_checks": adapter_checks,
        "notes": ["Known and unknown refer to membership in the provided course glossary, not all real-world knowledge.",
                  "Concept scoring is a preregistered keyword-group proxy; all raw outputs must also be reviewed.",
                  "The 58 terms occur in both train and valid; only question wording is held out.",
                  "The model receives no glossary definitions or index during inference; search reads the glossary directly.",
                  "Exploratory follow-up after the unsuccessful 5-epoch run: fixed 20 epochs, best checkpoint by valid loss every 40 steps.",
                  "The original 42 test outcomes were seen after the first run; the separate 12-question audit was fixed before this follow-up.",
                  "No unknown/refusal training examples; no data changes to train or valid between runs.",
                  "Data and operations assisted by Codex; no instructor result files used."],
    }
    save("result.json", result)
    manifest = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(HERE.rglob("*")) if p.is_file() and (p.parent == DATA or p.suffix == ".py")}
    save("run_manifest.json", manifest)
    print("ADVANCED_RESULT_READY", flush=True)
    print(f"training: {training_seconds:.2f} seconds / {trainer.state.global_step} steps", flush=True)
    for value in [before, after, search]:
        print(json.dumps(value["summary"], ensure_ascii=False), flush=True)
    names = ["result.json", "eval_before.json", "eval_after.json", "eval_search.json", "token_lengths.json", "run_manifest.json",
             "eval_audit_before.json", "eval_audit_after.json", "eval_audit_search.json"]
    payload = {name: (RESULTS / name).read_text(encoding="utf-8") for name in names}
    print("BEGIN_ADVANCED_RESULTS", flush=True)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    print("END_ADVANCED_RESULTS", flush=True)
    print(json.dumps({name: hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() for name in names}), flush=True)


if __name__ == "__main__":
    main()
