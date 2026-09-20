"""
Training Health Monitor — run every 2 hours to check that phase-3 training
has not diverged (loss → 0, NaN, or grad_norm explosion).

Usage:  D:/conda/envs/qwen/python.exe monitor_training.py
"""

import json
import os
import glob
import sys

LOGFILE = r"E:\model\Qwen3-4B-Base-finetuned\logs\training.log"
# Fallback if SFTConfig doesn't enable logging to file — read from trainer_state.json
CHECKPOINT_DIR = r"E:\model\Qwen3-4B-Base-finetuned"


def check_latest_checkpoint():
    """Scan the latest checkpoint for trainer_state.json and read recent loss."""
    checkpoints = sorted(
        glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint-*")),
        key=os.path.getmtime,
    )
    if not checkpoints:
        print("No checkpoints found yet.")
        return True  # not crashed, just early

    latest = checkpoints[-1]
    state_file = os.path.join(latest, "trainer_state.json")
    if not os.path.exists(state_file):
        print(f"No trainer_state.json in {latest}")
        return True

    with open(state_file, "r") as fh:
        state = json.load(fh)

    log_history = state.get("log_history", [])
    if not log_history:
        print("Empty log history.")
        return True

    # Check last 20 logged entries
    recent = log_history[-20:]
    anomalies = []
    for entry in recent:
        loss_val = entry.get("loss")
        grad_norm_val = entry.get("grad_norm")
        eval_loss_val = entry.get("eval_loss")

        if loss_val is not None:
            if loss_val != loss_val:  # NaN
                anomalies.append(f"NaN training loss at step {entry.get('step')}")
            elif loss_val < 0.01:
                anomalies.append(f"Near-zero training loss ({loss_val}) at step {entry.get('step')}")

        if grad_norm_val is not None and (grad_norm_val != grad_norm_val or grad_norm_val > 100):
            anomalies.append(f"Abnormal grad_norm ({grad_norm_val}) at step {entry.get('step')}")

        if eval_loss_val is not None and (eval_loss_val != eval_loss_val or eval_loss_val < 0.01):
            anomalies.append(f"Abnormal eval_loss ({eval_loss_val}) at step {entry.get('step')}")

    if anomalies:
        print("❌ ANOMALIES DETECTED:")
        for a in anomalies:
            print(f"   {a}")
        return False

    # Print healthy summary
    last_train = next((e for e in reversed(recent) if "loss" in e), None)
    last_eval = next((e for e in reversed(recent) if "eval_loss" in e), None)
    step = recent[-1].get("step", "?")
    epoch = recent[-1].get("epoch", "?")

    print(f"✅ Training healthy — step {step}, epoch {epoch}")
    if last_train:
        loss_v = last_train.get('loss')
        lr_v = last_train.get('learning_rate')
        loss_str = f"{loss_v:.4f}" if loss_v is not None else "N/A"
        lr_str = f"{lr_v:.2e}" if lr_v is not None else "N/A"
        print(f"   train_loss={loss_str}, lr={lr_str}")
    if last_eval:
        eval_v = last_eval.get('eval_loss')
        eval_str = f"{eval_v:.4f}" if eval_v is not None else "N/A"
        print(f"   eval_loss={eval_str}")

    # Estimate progress
    total_steps = state.get("max_steps", 6567)
    current = state.get("global_step", 0)
    if total_steps:
        pct = current / total_steps * 100
        remaining = total_steps - current
        # rough: 14s/step
        eta_h = remaining * 14 / 3600
        print(f"   Progress: {current}/{total_steps} ({pct:.1f}%) — ETA ~{eta_h:.1f}h remaining")

    return True


if __name__ == "__main__":
    ok = check_latest_checkpoint()
    print("\n" + "=" * 50)
    if ok:
        print("STATUS: HEALTHY")
    else:
        print("STATUS: CRASHED — INTERVENTION REQUIRED")
    print("=" * 50)
    sys.exit(0 if ok else 1)
