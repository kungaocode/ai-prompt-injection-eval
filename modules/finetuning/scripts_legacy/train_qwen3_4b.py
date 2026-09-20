"""
QLoRA Continued Pretraining for Qwen3-4B-Base
=============================================
Fine-tuning Qwen3-4B-Base on domain-specific CS/crypto/security corpora
using 4-bit quantization + LoRA adapters — optimized for RTX 4060 8 GB.

Dataset — 19,464 records, ~31 M chars, ~10 M estimated tokens
Format  — {"text": "..."}  (plain-text continued pretraining)

Usage:  D:/conda/envs/qwen/python.exe train_qwen3_4b.py
"""

import os
import json

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback
import datetime as _dt

# ── Anomaly detection callback ──────────────────────────────────────
class NaNGuardCallback(TrainerCallback):
    """Stop training on NaN/zero loss, and persist every log to a standalone file."""

    def __init__(self, log_path: str):
        self._log_path = log_path
        # write header (append mode to not overwrite existing data)
        if not os.path.exists(log_path):
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.write("step,epoch,loss,grad_norm,learning_rate,entropy,mean_token_accuracy,timestamp\n")

    def _write_entry(self, logs, step, tag="train"):
        """Persist one log entry to the loss log file (not affected by tqdm)."""
        row = [
            str(step),
            str(logs.get("epoch", "")),
            str(logs.get("loss", "")),
            str(logs.get("grad_norm", "")),
            str(logs.get("learning_rate", "")),
            str(logs.get("entropy", "")),
            str(logs.get("mean_token_accuracy", "")),
            _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        ]
        with open(self._log_path, "a", encoding="utf-8") as fh:
            fh.write(",".join(row) + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # Always persist
        self._write_entry(logs, state.global_step)

        loss = logs.get("loss")
        grad_norm = logs.get("grad_norm")
        if loss is not None and (loss != loss or loss <= 0.001):
            control.should_training_stop = True
        if grad_norm is not None and (grad_norm != grad_norm or grad_norm > 100):
            control.should_training_stop = True

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        self._write_entry(metrics, state.global_step, tag="eval")
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None and (eval_loss != eval_loss or eval_loss <= 0.001):
            control.should_training_stop = True

# ── Windows multiprocessing guard ─────────────────────────────────
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

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
SAVE_STEPS = 480        # ~2 hours between checkpoints (15s × 480 = 7200s)
SAVE_TOTAL_LIMIT = 8    # keep more rollback points
RESUME_CHECKPOINT = "E:/model/Qwen3-4B-Base-finetuned/checkpoint-2880"

# ── LoRA hyperparameters ───────────────────────────────────────────
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def load_and_prepare_dataset(data_dir: str, tokenizer):
    """Load all JSONL files, tokenize, return train/eval splits."""
    records = []
    total_chars = 0
    for root, _dirs, files in os.walk(data_dir):
        for fname in files:
            if not fname.endswith(".jsonl"):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        text = obj.get("text", "")
                        if text and len(text) > 50:
                            records.append({"text": text})
                            total_chars += len(text)
                    except json.JSONDecodeError:
                        continue

    dataset = Dataset.from_list(records)
    print(f"Loaded {len(dataset)} records, {total_chars:,} total chars")
    print(f"Estimated tokens: ~{int(total_chars / 3.5):,}")

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )

    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    # 90/10 split
    split = dataset.train_test_split(test_size=0.1, seed=42)
    return split["train"], split["test"]


def main():
    print("=" * 60)
    print("QLoRA Continued Pretraining: Qwen3-4B-Base")
    print(f"Model:   {MODEL_PATH}")
    print(f"Dataset: {DATASET_DIR}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    print(f"Torch: {torch.__version__}")
    print("=" * 60)

    # 1. Tokenizer
    print("\n[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load & tokenize dataset
    print("[2/3] Loading dataset...")
    train_dataset, eval_dataset = load_and_prepare_dataset(DATASET_DIR, tokenizer)
    print(f"Train: {len(train_dataset):,}, Eval: {len(eval_dataset):,}")

    # Run
    print("[3/3] Loading model + LoRA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.float16,
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # Resume from checkpoint (restores model + optimizer + scheduler + step)
    if os.path.isdir(RESUME_CHECKPOINT):
        print(f"Resuming from checkpoint: {RESUME_CHECKPOINT}")
        # Explicitly load PEFT adapter weights FIRST, so we don't rely on
        # Trainer._load_from_checkpoint to find adapter_model.safetensors.
        # Trainer still restores optimizer / scheduler / RNG / global_step.
        from safetensors.torch import load_file as _safetensors_load
        adapter_file = os.path.join(RESUME_CHECKPOINT, "adapter_model.safetensors")
        if os.path.isfile(adapter_file):
            adapter_state = _safetensors_load(adapter_file)
            model_state = model.state_dict()
            missing = [k for k in adapter_state if k not in model_state]
            if missing:
                print(f"WARNING: {len(missing)} adapter keys not found in model. First 5: {missing[:5]}")
            for name, param in adapter_state.items():
                if name in model_state:
                    model_state[name].copy_(param)
            print(f"Loaded {len(adapter_state) - len(missing)}/{len(adapter_state)} adapter weights")
        else:
            print(f"WARNING: adapter_model.safetensors not found in {RESUME_CHECKPOINT}")
        resume_from_checkpoint = RESUME_CHECKPOINT
    else:
        print(f"Checkpoint not found at {RESUME_CHECKPOINT}, starting fresh.")
        resume_from_checkpoint = None

    model.print_trainable_parameters()

    # 4. SFT Config (TRL >= 1.8)
    sft_config = SFTConfig(
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
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        fp16=False,
        bf16=False,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        dataloader_num_workers=0,
        report_to="none",
        seed=42,
        remove_unused_columns=False,
        save_only_model=False,
        load_best_model_at_end=False,
        dataset_text_field="text",
        packing=False,
        max_grad_norm=MAX_GRAD_NORM,
    )

    # 5. Trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,       # TRL >= 1.8 uses processing_class
        formatting_func=None,             # use dataset_text_field from config
        callbacks=[NaNGuardCallback(os.path.join(OUTPUT_DIR, "loss_log.csv"))],
    )

    print("\nStarting training...")
    if resume_from_checkpoint is not None:
        print(f"(Resuming from {resume_from_checkpoint})")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Total steps: {train_result.global_step}")
    print(f"Training loss: {train_result.training_loss:.4f}")
    print("=" * 60)

    # Save LoRA adapter
    adapter_path = os.path.join(OUTPUT_DIR, "lora-adapter")
    print(f"\nSaving LoRA adapter to {adapter_path}")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    # Merge & save full model (dequantizes from 4-bit → may use significant VRAM)
    merged_path = os.path.join(OUTPUT_DIR, "merged-model")
    print(f"Merging LoRA into base → {merged_path}")
    import gc as _gc
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)
    # Free merged-model VRAM before exiting
    del merged_model
    _gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("Outputs:")
    print(f"  LoRA adapter: {adapter_path}")
    print(f"  Merged model: {merged_path}")
    print(f"  Checkpoints:  {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
