"""
Diagnose problematic eval samples for checkpoint-6567.
Tests samples 25-49 individually, timing each, flagging outliers.
"""
import os, json, math, time, sys, gc
import torch

import transformers.utils.import_utils as _hf_iu
import transformers.trainer as _hf_trainer
_hf_iu.check_torch_load_is_safe = lambda: None
_hf_trainer.check_torch_load_is_safe = lambda: None

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import Dataset

MODEL_PATH = "E:/model/Qwen3-4B-Base"
DATASET_DIR = "E:/model/raw_for_training"
CKPT = "E:/model/Qwen3-4B-Base-finetuned/checkpoint-6567"
MAX_SEQ_LENGTH = 2048
BNB = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
)
TIME_THRESHOLD = 5.0   # seconds — sample taking longer than this is suspicious

def load_eval_samples(start, end):
    """Return eval samples [start:end] with their original indices."""
    records = []
    for root, _dirs, files in os.walk(DATASET_DIR):
        for fname in files:
            if not fname.endswith(".jsonl"):
                continue
            with open(os.path.join(root, fname), "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        text = obj.get("text", "")
                        if text and len(text) > 50:
                            records.append({"text": text})
                    except json.JSONDecodeError:
                        continue
    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    texts = split["test"][start:end]["text"]
    return texts


def main():
    print("=" * 60)
    print("Diagnosing problematic samples (indices 25-49)")
    print(f"Model: {CKPT}")
    print("=" * 60)

    # Load target samples
    print("\nLoading eval samples 25-49...")
    texts = load_eval_samples(25, 50)  # 25 inclusive, 50 exclusive
    print(f"Loaded {len(texts)} samples")

    # Also load some "good" samples for reference baseline (indices 0-5)
    print("Loading reference samples 0-5 for baseline...")
    ref_texts = load_eval_samples(0, 6)

    # Load model
    print("\nLoading model...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=BNB,
        device_map="auto", trust_remote_code=True,
        dtype=torch.float16, attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, CKPT)
    model.eval()

    # First, establish baseline time with 5 known-good samples
    print("\n--- Baseline (samples 0-5) ---")
    baseline_times = []
    for i, text in enumerate(ref_texts):
        enc = tok(text, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt")
        ids = enc["input_ids"].to(model.device)
        tokens = ids.size(1)

        t0 = time.time()
        with torch.no_grad():
            out = model(ids, labels=ids)
        elapsed = time.time() - t0

        baseline_times.append(elapsed)
        print(f"  sample {i:>3}: {tokens:>5} tokens, {elapsed:.2f}s, loss={out.loss.item():.4f}")

    avg_baseline = sum(baseline_times) / len(baseline_times)
    print(f"  Average baseline time: {avg_baseline:.2f}s")
    dynamic_threshold = max(avg_baseline * 5, TIME_THRESHOLD)
    print(f"  Dynamic threshold: {dynamic_threshold:.1f}s (5x baseline or {TIME_THRESHOLD}s, whichever larger)")

    # Now test the problematic range
    print(f"\n--- Probing samples 25-49 ---")
    results = []
    anomalies = []

    for i, text in enumerate(texts):
        real_idx = 25 + i
        enc = tok(text, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt")
        ids = enc["input_ids"].to(model.device)
        tokens = ids.size(1)

        print(f"  sample {real_idx:>3}: {tokens:>5} tokens... ", end="", flush=True)

        t0 = time.time()
        try:
            with torch.no_grad():
                out = model(ids, labels=ids)
            elapsed = time.time() - t0
            loss = out.loss.item()
        except Exception as e:
            elapsed = time.time() - t0
            loss = float("nan")
            print(f"ERROR after {elapsed:.1f}s: {e}", flush=True)
            anomalies.append((real_idx, tokens, elapsed, str(e), text[:200]))
            continue

        status = "OK"
        if elapsed > dynamic_threshold:
            status = "SLOW"
        if math.isnan(loss) or math.isinf(loss) or loss > 15 or loss < 0.01:
            status = "BAD_LOSS"

        print(f"{elapsed:.2f}s, loss={loss:.4f}  [{status}]", flush=True)

        results.append((real_idx, tokens, elapsed, loss, status, text[:200]))

        if status != "OK":
            anomalies.append((real_idx, tokens, elapsed, loss, text[:200]))

    # Summary
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print("=" * 60)

    all_times = [r[2] for r in results]
    avg_time = sum(all_times) / len(all_times)
    max_time = max(all_times)
    max_idx = [r for r in results if r[2] == max_time][0]

    print(f"\n{len(texts)} samples tested")
    print(f"Average time: {avg_time:.2f}s (baseline: {avg_baseline:.2f}s)")
    print(f"Slowest sample: idx={max_idx[0]}, {max_idx[2]:.2f}s, {max_idx[1]} tokens")
    print(f"Threshold used: {dynamic_threshold:.1f}s")

    if anomalies:
        print(f"\n{len(anomalies)} ANOMALIES FOUND:")
        for a in anomalies:
            idx, tokens, elapsed, extra, preview = a[0], a[1], a[2], a[3] if len(a) > 4 else "", a[-1]
            print(f"\n  --- Sample {idx} (idx in eval set) ---")
            print(f"  Tokens:  {tokens}")
            print(f"  Time:    {elapsed:.2f}s")
            if isinstance(extra, float):
                print(f"  Loss:    {extra:.4f}")
            else:
                print(f"  Error:   {extra}")
            print(f"  Preview: {preview[:300]}...")
    else:
        print("\nNo anomalies detected — all samples processed normally.")

    # Also do a rank by time
    print(f"\n--- Time ranking (all 25 samples) ---")
    ranked = sorted(results, key=lambda x: x[2], reverse=True)
    for rank, (idx, tokens, elapsed, loss, status, _) in enumerate(ranked, 1):
        bar = "#" * int(elapsed / max(all_times) * 40) if max(all_times) > 0 else ""
        flag = " <--" if status != "OK" else ""
        print(f"  {rank:>2}. sample {idx:>3}: {elapsed:>6.2f}s  {tokens:>5} tokens  loss={loss:.4f}  {bar}{flag}")

    del model, base, tok
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
