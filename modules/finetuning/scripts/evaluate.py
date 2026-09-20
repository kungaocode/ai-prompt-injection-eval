"""
Evaluate base model vs fine-tuned checkpoints on validation set.
Sequential 4-bit loading, one model at a time.
"""
import os, json, math, time, sys
import torch
import gc

import transformers.utils.import_utils as _hf_iu
import transformers.trainer as _hf_trainer
_hf_iu.check_torch_load_is_safe = lambda: None
_hf_trainer.check_torch_load_is_safe = lambda: None

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import Dataset

MODEL_PATH = "E:/model/Qwen3-4B-Base"
DATASET_DIR = "E:/model/raw_for_training"
FINETUNED_DIR = "E:/model/Qwen3-4B-Base-finetuned"
MAX_SEQ_LENGTH = 2048
EVAL_SAMPLES = 200

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
)

CHECKPOINTS = [
    ("base", None),
    ("ckpt-3840", os.path.join(FINETUNED_DIR, "checkpoint-3840")),
    ("ckpt-6567", os.path.join(FINETUNED_DIR, "checkpoint-6567")),
]


def load_eval_data():
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
    return split["test"]


def evaluate_one(name, ckpt_path, texts):
    print(f"  [1] Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  [2] Loading 4-bit model...", flush=True)
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=BNB_CONFIG,
        device_map="auto", trust_remote_code=True,
        dtype=torch.float16, attn_implementation="sdpa",
    )
    print(f"      Base loaded ({time.time()-t0:.0f}s | VRAM {torch.cuda.memory_allocated()/1e9:.1f}GB)", flush=True)

    if ckpt_path is not None:
        print(f"  [3] Loading adapter...", flush=True)
        t0 = time.time()
        model = PeftModel.from_pretrained(base, ckpt_path)
        print(f"      Adapter loaded ({time.time()-t0:.0f}s)", flush=True)
    else:
        model = base

    model.eval()
    print(f"  [4] Evaluating ({len(texts)} texts, max_len={MAX_SEQ_LENGTH})...", flush=True)

    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()

    with torch.no_grad():
        for i, text in enumerate(texts):
            enc = tokenizer(text, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt")
            input_ids = enc["input_ids"].to(model.device)
            if input_ids.size(1) < 2:
                continue
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
            n_tok = input_ids.size(1)
            total_loss += loss * n_tok
            total_tokens += n_tok

            if (i + 1) % 25 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 0.1)
                eta = max(0, (len(texts) - i - 1) / max(rate, 0.01))
                print(f"      {i+1}/{len(texts)} ({rate:.1f}/s | ETA {eta:.0f}s)", flush=True)

    elapsed = time.time() - t0
    ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    print(f"  => PPL={ppl:.4f}  Loss={avg_loss:.4f}  Tokens={total_tokens}  Time={elapsed:.0f}s", flush=True)

    del model, base, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

    return ppl, avg_loss


def main():
    print("=" * 60)
    print("Evaluation: base vs ckpt-3840 (best) vs ckpt-6567 (final)")
    print(f"Samples: {EVAL_SAMPLES}  |  Max sequence length: {MAX_SEQ_LENGTH}")
    print("=" * 60)

    print()
    print("Loading eval data...")
    eval_texts = load_eval_data()[:EVAL_SAMPLES]["text"]
    print(f"Loaded {len(eval_texts)} samples")
    print()

    results = {}
    for name, ckpt_path in CHECKPOINTS:
        print("=" * 50)
        print(f"  Model: {name}")
        print("=" * 50)
        ppl, loss = evaluate_one(name, ckpt_path, eval_texts)
        results[name] = {"ppl": ppl, "loss": loss}
        print()

    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f'{"Model":<15} {"PPL":>10} {"AvgLoss":>10} {"vs Base":>12}')
    print("-" * 50)
    base_ppl = results["base"]["ppl"]
    for name in ["base", "ckpt-3840", "ckpt-6567"]:
        ppl = results[name]["ppl"]
        loss = results[name]["loss"]
        vs_base = (ppl - base_ppl) / base_ppl * 100
        print(f'{name:<15} {ppl:>10.4f} {loss:>10.4f} {vs_base:>+11.2f}%')

    best_ppl = min(r["ppl"] for r in results.values())
    print()
    print(f"Best improvement over base: {(base_ppl - best_ppl) / base_ppl * 100:.2f}%")


if __name__ == "__main__":
    main()
