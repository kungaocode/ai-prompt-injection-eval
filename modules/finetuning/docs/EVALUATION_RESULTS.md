=== FINE-TUNING EVALUATION RESULTS ===
Date: 2026-08-09
Model: Qwen3-4B-Base -> QLoRA domain continued pretraining
Domains: Computer Science, Cryptography & Security
Data: 19,452 records, ~8.9M tokens

--- 2048 seq, 175 validation samples (skip 25-49 problematic) ---

Model                         PPL       Loss      vs Base
--------------------------------------------------------------
base (original Qwen3-4B)    12.3144   2.5108    baseline
ckpt-3840 (best eval)        9.5594   2.2575    -22.4%
ckpt-6567 (final, epoch 3)  10.2757   2.3298    -16.6%

--- 512 seq, 100 validation samples (all consistent) ---

base (original Qwen3-4B)    14.0573   2.6431    baseline
ckpt-3840 (best eval)       10.0857   2.3111    -28.3%
ckpt-6567 (final, epoch 3)  10.7876   2.3784    -23.3%

Key findings:
1. Fine-tuning is EFFECTIVE: domain PPL reduced 22-28%
2. Best checkpoint: checkpoint-3840 (epoch ~1.75, eval_loss=2.2334)
3. Overfit confirmed: final model (epoch 3) degrades 7-8% vs best
4. Recommended for downstream use: checkpoint-3840
