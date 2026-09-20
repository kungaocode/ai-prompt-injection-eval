# Qwen3-4B-Base QLoRA 微调参数说明与训练流程文档

## 概述

本项目使用 **QLoRA (Quantized Low-Rank Adaptation)** 技术对 **Qwen3-4B-Base** 基础模型进行继续预训练 (Continued Pretraining)，在计算机科学、密码学与信息安全等领域的专业语料上进行领域适配。训练针对 **NVIDIA RTX 4060 8 GB** 显存进行了优化。

---

## 一、基础模型信息 (`config.json`)

基础模型为 Qwen3-4B-Base，权重格式为 `safetensors`（分片为 3 个文件，共计约 7.8 GB）。以下为核心架构参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| `architectures` | `["Qwen3ForCausalLM"]` | 模型架构类型，因果语言模型（自回归） |
| `model_type` | `qwen3` | Hugging Face Transformers 注册的模型类型标识 |
| `hidden_size` | 2560 | 隐藏层维度，即每个 token 的嵌入向量维度 |
| `num_hidden_layers` | 36 | Transformer 层数（深度） |
| `num_attention_heads` | 32 | 查询 (Query) 注意力头数 |
| `num_key_value_heads` | 8 | 键/值 (Key/Value) 注意力头数，使用 GQA (Grouped Query Attention)，4 个 Q 头共享 1 组 KV 头 |
| `head_dim` | 128 | 每个注意力头的维度 |
| `intermediate_size` | 9728 | FFN (前馈网络) 中间层维度 |
| `hidden_act` | `silu` | 激活函数，SiLU (Swish) |
| `vocab_size` | 151936 | 词表大小 |
| `max_position_embeddings` | 32768 | 最大位置编码长度，支持 32K 上下文 |
| `rope_theta` | 1000000 | RoPE (旋转位置编码) 的基础频率 |
| `rms_norm_eps` | 1e-06 | RMS LayerNorm 的 epsilon 值，防止除零 |
| `tie_word_embeddings` | `true` | 输入嵌入层与输出投影层共享权重 |
| `torch_dtype` | `bfloat16` | 原始模型的数据类型为 BF16 |
| `bos_token_id` / `eos_token_id` | 151643 | 开始/结束 token（均为 `<|endoftext|>`） |
| `initializer_range` | 0.02 | 参数初始化时的标准差范围 |
| `attention_dropout` | 0.0 | 注意力 dropout 率（预训练时通常为 0） |
| `max_window_layers` | 36 | 使用滑动窗口注意力的最大层数（此处为全部层） |
| `use_sliding_window` | `false` | 当前未启用滑动窗口注意力 |

---

## 二、训练脚本参数详解 (`train_qwen3_4b.py`)

### 2.1 路径配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `MODEL_PATH` | `E:/model/Qwen3-4B-Base` | 预训练基础模型的本地路径 |
| `DATASET_DIR` | `E:/model/raw_for_training` | 训练数据根目录，包含多个子目录和 JSONL 文件 |
| `OUTPUT_DIR` | `E:/model/Qwen3-4B-Base-finetuned` | 微调后模型的输出目录 |

### 2.2 训练超参数 (Training Hyperparameters)

| 参数 | 值 | 说明 |
|------|-----|------|
| `MAX_SEQ_LENGTH` | 2048 | 最大序列长度（token 数）。超过此长度的文本将被截断。选择 2048 以在显存与上下文覆盖之间取得平衡 |
| `BATCH_SIZE` | 1 | 每张 GPU 的训练批次大小。由于 8 GB 显存受限设为 1 |
| `GRADIENT_ACCUMULATION` | 8 | 梯度累积步数。等效批次大小 = `BATCH_SIZE × GRADIENT_ACCUMULATION = 8`。每 8 个 micro-batch 后才更新一次权重，在小显存下模拟更大的批次 |
| `LEARNING_RATE` | 2e-4 | 学习率。QLoRA 论文建议 LoRA 微调可用比全参数微调更高的学习率，2e-4 是常用起点 |
| `WARMUP_STEPS` | 50 | 预热步数。训练开始时的 50 步内，学习率从 0 线性增长到 `LEARNING_RATE`，有助于训练稳定性 |
| `NUM_EPOCHS` | 3 | 训练轮数。整个数据集完整遍历 3 次 |
| `LOGGING_STEPS` | 10 | 每 10 步记录一次训练日志（loss 等指标） |
| `SAVE_STEPS` | 200 | 每 200 步保存一次中间检查点 |
| `SAVE_TOTAL_LIMIT` | 3 | 最多保留 3 个检查点，旧的将被自动删除以节省磁盘空间 |

### 2.3 LoRA 超参数 (LoRA Hyperparameters)

LoRA 通过在权重矩阵中插入低秩分解矩阵来实现参数高效微调，只训练少量额外参数。

| 参数 | 值 | 说明 |
|------|-----|------|
| `LORA_R` | 16 | LoRA 的秩 (rank)。低秩分解矩阵的维度。r 越大，适配器容量越大，但参数量也越多。16 是常用值 |
| `LORA_ALPHA` | 32 | LoRA 的缩放因子。实际缩放比为 `alpha / r = 32 / 16 = 2`。增大 alpha 会放大 LoRA 权重的影响 |
| `LORA_DROPOUT` | 0.05 | LoRA 层的 dropout 率，5% 的随机丢弃用于防止过拟合 |
| `TARGET_MODULES` | 见下方 | 应用 LoRA 的目标模块列表 |

**目标模块详解：**

| 模块 | 所属组件 | 说明 |
|------|---------|------|
| `q_proj` | Self-Attention | 查询 (Query) 投影矩阵 |
| `k_proj` | Self-Attention | 键 (Key) 投影矩阵 |
| `v_proj` | Self-Attention | 值 (Value) 投影矩阵 |
| `o_proj` | Self-Attention | 输出 (Output) 投影矩阵 |
| `gate_proj` | FFN (MLP) | 门控投影矩阵（SwiGLU 的门控分支） |
| `up_proj` | FFN (MLP) | 上投影矩阵（SwiGLU 的上采样分支） |
| `down_proj` | FFN (MLP) | 下投影矩阵 |

> 对 Self-Attention 的 Q/K/V/O 和 FFN 的所有线性层都应用 LoRA，实现了全面适配。

### 2.4 量化配置 (BitsAndBytes QLoRA)

| 参数 | 值 | 说明 |
|------|-----|------|
| `load_in_4bit` | `true` | 启用 4-bit 量化加载。将模型权重压缩到 4-bit，大幅减少显存占用 |
| `bnb_4bit_quant_type` | `"nf4"` | 量化数据类型为 NormalFloat4。这是 QLoRA 论文提出的信息论最优 4-bit 数据类型，专为正态分布的权重设计 |
| `bnb_4bit_compute_dtype` | `torch.float16` | 计算时反量化到 FP16 精度。保持前向/反向传播的计算精度 |
| `bnb_4bit_use_double_quant` | `true` | 启用双重量化。对量化常数本身再进行一次量化，进一步节省约 0.4 字节/参数 |

### 2.5 SFT 训练配置 (SFTConfig)

| 参数 | 值 | 说明 |
|------|-----|------|
| `output_dir` | `OUTPUT_DIR` | 训练输出目录 |
| `per_device_train_batch_size` | 1 | 每设备训练批次大小 |
| `per_device_eval_batch_size` | 1 | 每设备验证批次大小 |
| `gradient_accumulation_steps` | 8 | 梯度累积步数 |
| `learning_rate` | 2e-4 | 学习率 |
| `warmup_steps` | 50 | 学习率预热步数 |
| `num_train_epochs` | 3 | 训练轮数 |
| `logging_steps` | 10 | 日志记录间隔 |
| `save_steps` | 200 | 检查点保存间隔 |
| `save_total_limit` | 3 | 最多保留检查点数 |
| `eval_strategy` | `"steps"` | 验证策略：按步数触发评估 |
| `eval_steps` | 200 | 每 200 步进行一次验证 |
| `fp16` / `bf16` | `false` / `false` | 禁用 FP16/BF16 混合精度。因为基础模型已是 4-bit 量化 + FP16 计算精度，避免冲突 |
| `lr_scheduler_type` | `"cosine"` | 学习率调度器：余弦退火。学习率按余弦曲线从初始值衰减到接近 0，使训练末期更平滑 |
| `optim` | `"adamw_torch"` | 优化器：AdamW (PyTorch 内置实现)。Adam 的改进版，权重衰减与学习率解耦 |
| `dataloader_num_workers` | 0 | 数据加载的子进程数。Windows 下设为 0 以避免多进程问题 |
| `report_to` | `"none"` | 不上报训练日志到外部平台（如 Wandb/TensorBoard） |
| `seed` | 42 | 随机种子，确保可复现性 |
| `remove_unused_columns` | `false` | 不移除未使用的列。继续预训练中所有数据列都参与训练 |
| `save_only_model` | `false` | 保存完整训练状态（优化器、调度器等），便于恢复训练，而不仅是模型权重 |
| `load_best_model_at_end` | `false` | 训练结束时不加载最佳模型。由于继续预训练的 loss 不一定与下游任务表现相关，选择使用最终模型 |
| `dataset_text_field` | `"text"` | 数据集中的文本字段名。训练器将从此字段读取文本进行训练 |
| `packing` | `false` | 不启用序列打包。每条样本独立处理，不足 max_length 的部分用 pad token 填充 |

### 2.6 模型加载配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `device_map` | `"auto"` | 自动将模型各层分配到可用设备（GPU/CPU），最大化显存利用 |
| `trust_remote_code` | `true` | 信任模型仓库中的自定义代码（Qwen3 需要自定义 modeling 代码） |
| `attn_implementation` | `"sdpa"` | 注意力实现方式：PyTorch 原生 SDPA (Scaled Dot-Product Attention)，比传统实现更快、更省显存 |
| `gradient_checkpointing_enable` | `true` | 启用梯度检查点。用计算换显存：不保存所有中间激活值，反向传播时重新计算，大幅节省显存 |
| `bias` | `"none"` | LoRA 不训练 bias 参数，只训练低秩矩阵 |
| `task_type` | `TaskType.CAUSAL_LM` | 任务类型：因果语言模型（自回归预测下一个 token） |

---

## 三、训练数据集

### 3.1 数据格式

数据以 JSONL 格式存储，每行一个 JSON 对象，包含 `text` 字段：

```json
{"text": "训练文本内容..."}
```

### 3.2 数据目录结构

```
raw_for_training/
├── comprehensive/
│   ├── cross_domain.jsonl          # 跨领域综合知识
│   └── general.jsonl               # 通用知识（信息安全标准文档等）
├── computer_science/
│   ├── algorithms_and_data_structures.jsonl   # 算法与数据结构
│   ├── artificial_intelligence.jsonl          # 人工智能
│   ├── computer_architecture.jsonl            # 计算机体系结构
│   ├── computer_networks.jsonl                # 计算机网络
│   ├── databases.jsonl                        # 数据库
│   ├── operating_systems.jsonl                # 操作系统
│   ├── programming_languages_and_se.jsonl     # 编程语言与软件工程
│   ├── theory_and_discrete_math.jsonl         # 理论计算机科学与离散数学
│   └── data.jsonl                             # 计算机科学综合数据
└── crypto_and_security/
    ├── authentication_and_access_control.jsonl    # 认证与访问控制
    ├── cryptographic_algorithms.jsonl             # 密码算法
    ├── cryptographic_foundations.jsonl            # 密码学基础
    ├── cryptographic_protocols.jsonl              # 密码协议
    ├── key_management_and_pki.jsonl               # 密钥管理与公钥基础设施
    ├── network_security.jsonl                     # 网络安全
    ├── security_engineering_and_governance.jsonl  # 安全工程与治理
    ├── system_security.jsonl                      # 系统安全
    ├── web_security.jsonl                         # Web 安全
    └── data.jsonl                                 # 密码与安全综合数据
```

### 3.3 数据统计

| 指标 | 值 |
|------|-----|
| 总记录数 | 19,464 条 |
| 总字符数 | ~31,000,000 |
| 估算 Token 数 | ~10,000,000 (按 3.5 字符/token 估算) |

### 3.4 数据预处理

1. **加载**: 递归遍历 `DATASET_DIR` 下所有 `.jsonl` 文件
2. **过滤**: 丢弃空文本或长度 ≤ 50 字符的短文本
3. **分词**: 使用 Qwen3 的 tokenizer，将文本转换为 token IDs，截断到 `max_length=2048`
4. **划分**: 90% 训练集 / 10% 验证集（`seed=42`）

---

## 四、完整微调流程

```
┌──────────────────────────────────────────────────────────────────┐
│  步骤 1: 加载 Tokenizer                                         │
│  • 从 MODEL_PATH 加载 Qwen3 tokenizer                           │
│  • 若 pad_token 缺失，将 eos_token 设为 pad_token                │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  步骤 2: 加载并预处理数据集                                      │
│  • 递归读取 raw_for_training/**/*.jsonl 文件                     │
│  • JSON 解析 → 过滤短文本 → 分词 → 90/10 划分                    │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  步骤 3: 加载模型 + QLoRA 配置                                   │
│  • 4-bit 量化加载基础模型 (BitsAndBytesConfig)                   │
│  • 设备映射: device_map="auto"                                   │
│  • 为 k-bit 训练准备模型 (prepare_model_for_kbit_training)       │
│  • 启用梯度检查点 (gradient_checkpointing_enable)                │
│  • 配置 LoRA 适配器 (LoraConfig)                                 │
│  • 获取 PEFT 模型 (get_peft_model)                               │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  步骤 4: 配置训练器                                              │
│  • SFTConfig: 批次/学习率/调度器/验证/保存策略                    │
│  • SFTTrainer: 封装模型、数据集、分词器                           │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  步骤 5: 开始训练                                                │
│  • 每个 step: 前向 → 反向 → 梯度累积(×8) → 优化器更新            │
│  • 每 10 steps: 记录 loss                                       │
│  • 每 200 steps: 验证 → 保存检查点                               │
│  • 学习率: warmup(50步) → cosine 衰减到 ~0                       │
│  • 训练 3 个 epoch                                               │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  步骤 6: 保存模型                                                │
│  • 保存 LoRA 适配器权重 (轻量，仅含训练的参数)                    │
│  • 合并 LoRA + 基础模型 → 保存完整模型                           │
│  • 保存 tokenizer                                                │
└──────────────────────────────────────────────────────────────────┘
```

### 输出文件

| 输出 | 路径 | 说明 |
|------|------|------|
| LoRA 适配器 | `OUTPUT_DIR/lora-adapter/` | 仅包含训练的 LoRA 权重，文件小，加载时需要配合基础模型 |
| 合并模型 | `OUTPUT_DIR/merged-model/` | LoRA 与基础模型合并后的完整权重，可直接用于推理 |
| 检查点 | `OUTPUT_DIR/checkpoint-*/` | 训练中间检查点，包含优化器状态，可用于恢复训练 |

---

## 五、关键设计决策与原理

### 5.1 为什么使用 QLoRA 而不是全参数微调？

Qwen3-4B 约有 40 亿参数，全参数 FP16 微调需要约 8 GB 仅用于模型权重，加上优化器状态（AdamW 需要额外 2 份动量参数）和激活值，总显存需要超过 32 GB。QLoRA 通过 4-bit 量化将模型权重压缩到约 2 GB，加上 LoRA 仅训练少量参数（~1-2% 的原始参数量），使得在 8 GB 显存的 RTX 4060 上也能完成微调。

### 5.2 为什么选择 `r=16, alpha=32`？

- `r=16` 提供了足够的适配容量来学习领域特定知识，同时参数量可控
- `alpha=32` 使缩放比为 2.0，为 LoRA 权重提供了合理的信号放大

### 5.3 为什么用 Cosine 学习率调度？

余弦退火 (`cosine`) 在训练末期学习率平滑衰减到接近 0，相比线性衰减更能避免末期震荡，有助于收敛到更好的局部最优解。

### 5.4 为什么使用 SDPA 注意力？

PyTorch 原生的 `torch.nn.functional.scaled_dot_product_attention` (SDPA) 会根据输入自动选择最优的注意力实现（Flash Attention 2、Memory-Efficient Attention 或标准实现），在速度和显存占用上均优于传统的手动实现。

### 5.5 Multi-processing 处理

代码中特别设置了 `multiprocessing.freeze_support()` 和 `spawn` 启动方法，这是因为 Windows 下 Python 的默认 `fork` 模式不可用，需要使用 `spawn` 来避免子进程初始化时的死锁和资源冲突。同时 `dataloader_num_workers=0` 也是在 Windows 下的兼容性设置。

### 5.6 继续预训练 vs 指令微调

本项目采用继续预训练 (Continued Pretraining) 而非指令微调 (Instruction Fine-tuning)：
- **数据格式**: 纯文本 `{"text": "..."}`，不包含对话格式或指令模板
- **目标**: 让模型学习计算机科学、密码学、安全领域的知识与语言模式，而非学习对话格式
- **结果**: 微调后的模型仍然是基础模型，可用于后续的指令微调或直接用于特定领域任务
