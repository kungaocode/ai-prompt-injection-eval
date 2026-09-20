"""
QLoRA Continued Pretraining for Qwen3-4B-Base — RESUME VERSION
===============================================================
Resumes from the latest checkpoint found in OUTPUT_DIR.
Restores adapter weights + optimizer state + scheduler state + global_step.

Usage:  D:/conda/envs/qwen/python.exe resume_train.py
"""
import time as _time
import os
import json
import glob

# ── Patch transformers + torch 2.5.1 compatibility BEFORE any other imports ──
import transformers.utils.import_utils as _hf_iu
import transformers.trainer as _hf_trainer
_hf_iu.check_torch_load_is_safe = lambda: None
_hf_trainer.check_torch_load_is_safe = lambda: None

import torch
_orig_torch_load = torch.load
def _safe_load(*a, **kw):
    kw["weights_only"] = False
    return _orig_torch_load(*a, **kw)
torch.load = _safe_load

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTConfig, SFTTrainer

# ── Paths ──────────────────────────────────────────────────────────
MODEL_PATH = "E:/model/Qwen3-4B-Base"
DATASET_DIR = "E:/model/raw_for_training"
OUTPUT_DIR = "E:/model/Qwen3-4B-Base-finetuned"

# ── Training hyperparameters ───────────────────────────────────────
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 2.5e-5
WARMUP_STEPS = 0       # resume: warmup already completed
NUM_EPOCHS = 3
MAX_GRAD_NORM = 0.3
LOGGING_STEPS = 5
SAVE_STEPS = 480
SAVE_TOTAL_LIMIT = 20

# ── LoRA hyperparameters ───────────────────────────────────────────
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Return the path to the highest-step checkpoint, or None."""
    pattern = os.path.join(output_dir, "checkpoint-*")
    candidates = sorted(
        glob.glob(pattern),
        key=lambda p: int(os.path.basename(p).split("-")[-1]),
    )
    if not candidates:
        return None
    return candidates[-1]


# ── Anomaly detection callback ──────────────────────────────────────
class GuardCallback(TrainerCallback):
    """Stop training on NaN/zero loss, and persist every log to a standalone file."""

    def __init__(self, log_path: str):
        self._path = log_path
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("step,epoch,loss,grad_norm,learning_rate,entropy,mean_token_accuracy,timestamp\n")

    def _write(self, logs, step):
        import datetime
        vals = [
            str(step), str(logs.get("epoch", "")),
            str(logs.get("loss", "")), str(logs.get("grad_norm", "")),
            str(logs.get("learning_rate", "")), str(logs.get("entropy", "")),
            str(logs.get("mean_token_accuracy", "")),
            datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        ]
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(",".join(vals) + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        self._write(logs, state.global_step)
        loss = logs.get("loss")
        gn = logs.get("grad_norm")
        if loss is not None and (loss != loss or loss <= 0.001):
            control.should_training_stop = True
        if gn is not None and (gn != gn or gn > 100):
            control.should_training_stop = True

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        self._write(metrics, state.global_step)
        el = metrics.get("eval_loss")
        if el is not None and (el != el or el <= 0.001):
            control.should_training_stop = True


def load_data(data_dir, tokenizer):
    records = []
    tc = 0
    for root, _dirs, files in os.walk(data_dir):
        for fn in files:
            if not fn.endswith(".jsonl"):
                continue
            with open(os.path.join(root, fn), "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        t = obj.get("text", "")
                        if t and len(t) > 50:
                            records.append({"text": t})
                            tc += len(t)
                    except json.JSONDecodeError:
                        continue
    ds = Dataset.from_list(records)
    print(f"Loaded {len(ds)} records, {tc:,} chars")
    ds = ds.map(lambda x: tokenizer(x["text"], truncation=True, max_length=MAX_SEQ_LENGTH),
                batched=True, remove_columns=["text"], desc="Tokenizing")
    sp = ds.train_test_split(test_size=0.1, seed=42)
    return sp["train"], sp["test"]


def main():
    # ── Auto-detect latest checkpoint ───────────────────────────────
    resume_checkpoint = find_latest_checkpoint(OUTPUT_DIR)
    if not resume_checkpoint:
        raise FileNotFoundError(f"No checkpoint found in {OUTPUT_DIR}. Run train_qwen3_4b.py first.")
    adapter_file = os.path.join(resume_checkpoint, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        raise FileNotFoundError(f"adapter_model.safetensors not found in {resume_checkpoint}")

    # Read trainer_state to show resume status
    state_file = os.path.join(resume_checkpoint, "trainer_state.json")
    if os.path.exists(state_file):
        with open(state_file, "r") as fh:
            prev_state = json.load(fh)
        resume_step = prev_state.get("global_step", 0)
        resume_epoch = prev_state.get("epoch", 0)
        max_steps = prev_state.get("max_steps", "?")
        print(f"Resume checkpoint: {resume_checkpoint}")
        print(f"  global_step = {resume_step}")
        print(f"  epoch       = {resume_epoch:.4f}")
        if isinstance(max_steps, int):
            remaining = max_steps - resume_step
            eta_h = remaining * 14 / 3600
            print(f"  max_steps   = {max_steps}")
            print(f"  remaining   = {remaining} steps (~{eta_h:.1f}h at ~14s/step)")

    # ── Timestamped loss log ────────────────────────────────────────
    loss_log = os.path.join(OUTPUT_DIR, f"loss_log_{_time.strftime('%Y%m%d_%H%M%S')}.csv")

    print("=" * 60)
    print("QLoRA Resume Training: Qwen3-4B-Base")
    print(f"Model:     {MODEL_PATH}")
    print(f"Resume from: {resume_checkpoint}")
    print(f"GPU:       {torch.cuda.get_device_name(0)}")
    print(f"VRAM:      {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    print(f"Torch:     {torch.__version__}")
    print(f"Loss log:  {loss_log}")
    print("=" * 60)

    # 1. Tokenizer
    print("\n[1/3] Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 2. Dataset
    print("[2/3] Loading dataset...")
    train_ds, eval_ds = load_data(DATASET_DIR, tok)
    print(f"Train: {len(train_ds):,}, Eval: {len(eval_ds):,}")

    # 3. Model + LoRA
    print("[3/3] Loading model + LoRA...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb,
        device_map="auto", trust_remote_code=True,
        dtype=torch.float16, attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)

    # NOTE: adapter weights restored by Trainer via resume_from_checkpoint.
    model.print_trainable_parameters()

    # 4. SFT Config
    sft_cfg = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        fp16=False, bf16=False,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        dataloader_num_workers=0,
        report_to="none", seed=42,
        remove_unused_columns=False,
        save_only_model=False,
        load_best_model_at_end=False,
        dataset_text_field="text",
        packing=False,
        max_grad_norm=MAX_GRAD_NORM,
    )

    # 5. Trainer
    trainer = SFTTrainer(
        model=model, args=sft_cfg,
        train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=tok,
        callbacks=[GuardCallback(loss_log)],
    )

    print(f"\nLoss log: {loss_log}")
    print(f"\nStarting training - RESUMING FULL STATE from {resume_checkpoint}:")
    print("  - adapter weights  -- loaded from checkpoint by Trainer")
    print("  - optimizer state  -- AdamW momentum restored from checkpoint")
    print("  - scheduler state  -- cosine LR curve continued")
    print("  - global_step      -- restored from trainer_state.json")
    print("  - rng state        -- random seed state restored")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Total steps: {train_result.global_step}")
    print(f"Training loss: {train_result.training_loss:.4f}")
    print("=" * 60)

    # Save outputs
    ap = os.path.join(OUTPUT_DIR, "lora-adapter")
    print(f"Saving LoRA adapter -> {ap}")
    trainer.model.save_pretrained(ap)
    tok.save_pretrained(ap)

    mp = os.path.join(OUTPUT_DIR, "merged-model")
    print(f"Merging -> {mp}")
    import gc as _gc
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(mp, safe_serialization=True)
    tok.save_pretrained(mp)
    del merged
    _gc.collect()
    torch.cuda.empty_cache()

    print(f"\nDone. LoRA: {ap}  |  Merged: {mp}  |  Loss log: {loss_log}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
