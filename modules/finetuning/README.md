# QLoRA 微调模块(已整理)

> Qwen3-4B-Base 在 CS/密码学/信息安全领域的 QLoRA 领域继续预训练。
> 关键结论:最优检查点是 **ckpt-3840**(epoch 1.75,PPL 较 Base −22~28%),
> 最终 ckpt-6567(epoch 3.0)已过拟合、劣化 7~8%。

## 目录

| 目录 | 内容 |
|---|---|
| `scripts/` | **最终可用版**脚本(含 6 个 bug 修复):`train_qwen3_4b.py`、`resume_train.py`、`monitor_training.py`、`evaluate.py`、`diagnose_samples.py`、`write_report.py` |
| `scripts_legacy/` | 早期版脚本(2026-08-07,修复前),仅作版本存档 |
| `reports/` | 三阶段报告(phase1/2/3 的 tex+pdf)与 `security_degradation_experiment_plan.md`(第 4 阶段 garak 实验计划) |
| `logs/` | 训练 loss 日志与健康检查快照 |
| `docs/` | 各阶段说明、评估结果、完整报告、检查点索引 |
| `model_assets/` | 模型 config / 分词器 / 词表 / 许可证 / 权重索引 |

## 说明

- 训练/评估环境是 **Windows**(GBK 代码页、torch 2.5.1+cu121、transformers 5.13.1),与当前评测代码(云 API,无需 GPU)无关。
- `reports/security_degradation_experiment_plan.md` 提出用三检查点(Base / ckpt-3840 / ckpt-6567)做"过拟合→安全性退化"的剂量-效应实验——该计划原用 garak,若改用本仓库的 `src/` 自研流水线,可直接复用这三个模型作为被测对象。
