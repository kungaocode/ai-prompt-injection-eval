# LLM Safety Evaluation: Prompt Injection & Guardrails

> CV 项目 02。**自研** prompt-injection 评测流程(不依赖 garak):自己定义探针、判定口径、防御与指标。
> 被测对象全部走**云 API**(OpenAI 兼容,默认阿里云百炼 DashScope),免本地部署成本。

## 目录结构

```
├── 立项书.md             # 项目计划(目标/范围/时间表)
├── taxonomy.md           # 探针分类学 + 判定口径(预注册)
├── Makefile              # make all / make mock / make test
├── configs/
│   └── models.yaml       # 云模型端点 + 版本 + secret + 系统提示模板
├── probes/
│   ├── generate_probes.py  # 探针生成器(模板 → JSONL)
│   ├── en/*.jsonl          # 英文探针(按类别分文件)
│   └── zh/*.jsonl          # 中文探针
├── benign/
│   └── benign_inputs.jsonl # 良性输入集(算 FPR)
├── src/
│   ├── runner.py           # 云 API 调用 + 离线 mock
│   ├── judge.py            # 确定性判定 + LLM judge
│   ├── guard.py            # instruction-hierarchy / input-filter / output-check
│   ├── report.py           # ASR / FPR 汇总
│   └── main.py             # 入口
├── results/                # 原始输出 + 报告(运行时生成)
└── modules/
    ├── finetuning/         # QLoRA 微调模块(已整理)
    └── garak-baseline/     # 之前的 garak 基线实验(参考)
```

## 快速开始

```bash
# 1. 配置 API key
cp .env.example .env        # 填入 DASHSCOPE_API_KEY

# 2. 安装依赖
pip install -r requirements.txt

# 3. 离线跑通(无需 key,假模型验证流水线)
make mock

# 4. 真实云 API 跑基线
make run

# 5. 带 guard 的 2×2 复测 + 良性 FPR(当前为 mock)
make test
```

产出在 `results/report.md`(`records.jsonl` / `judge_results.jsonl` 为可复现原始数据)。

## 方法一句话

给系统一个需保护的 secret + 使用规则;5 类注入探针(中英各一套)诱导模型泄露 secret;
**确定性规则**判定是否泄露(零主观)→ 加 guard 后复测 → 报告 **ASR 下降 + 良性 FPR 上升**,
论证"防御不是免费的"。

## 状态

- [x] `modules/finetuning/` — Qwen3-4B-Base QLoRA 领域持续预训练(PPL −22~28%)
- [x] `modules/garak-baseline/` — Garak 基线(社工越狱绕过 96.67% / 数据泄露 6.67% / 注入 0%)
- [x] 自研评测基础模块(probes / runner / judge / guard / report,云 API)
- [ ] 真实云 API 跑出基线 ASR(需 API key)
- [ ] guard 2×2 复测 + 良性 FPR
- [ ] 人工抽审一致性
- [ ] 英文 README + 结果表 + 推 GitHub

> 最终 GitHub README 用英文;本文为工作文档。
