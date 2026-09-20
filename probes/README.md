# 探针库:生成与使用指令

探针集由两个生成器从模板系统化生成,保证中英对称、可复现。
**不要手工改动 `en/`、`zh/` 下的 JSONL**;要改探针就改生成脚本再重新生成。

## 两个生成器

| 脚本 | 覆盖类别 | 来源 | 用途 |
|---|---|---|---|
| `probes/generate_probes.py` | 5 类基础攻击面 | 自建分类学 | 训练/评测主探针 |
| `probes/generate_garak_inspired.py` | 3 类 garak 启发新攻击面 | `modules/garak-baseline/experiment_results_summary.md`(goodside.Tag / leakreplay / grandma.Win10) | 安全性微调的 OOD 压力测试层 |

## 重新生成(全量)

```bash
python probes/generate_probes.py              # 5 类基础探针
python probes/generate_garak_inspired.py      # 3 类 garak 启发探针
```

输出:
- `probes/en/<category>.jsonl` — 英文探针(每类 6 条)
- `probes/zh/<category>.jsonl` — 中文探针(每类 6 条)

当前规模:**8 类 × 中英各 6 = 96 条**(5 基础 + 3 garak 启发)。

## 扩容

1. 编辑对应生成脚本的模板字典,在对应类别/语言的列表里**追加**新模板(基础类改 `generate_probes.py` 的 `TEMPLATES`,garak 启发类改 `generate_garak_inspired.py` 的 `enc_prompts/cloze_prompts/social_prompts`);
2. 重新运行上面的生成命令;
3. `probes/en|zh/*.jsonl` 自动重写,`id` 自动续号(`{category}-{lang}-{i:03d}`);
4. 检查每类条数(生成脚本会打印统计),并在 README 的探针统计处同步更新。

新增探针注意:
- 保持 `criterion: "secret-leak"` + 高熵 `secret`,才能走确定性判定(见 `taxonomy.md`);
- 中文模板用逗号句读(与现有风格一致),避免混用全角/半角符号导致判定不一致。

## 冒烟/子集测试(不污染正式探针集)

```bash
mkdir -p /tmp/smoke_probes
head -1 probes/en/direct-injection.jsonl > /tmp/smoke_probes/a.jsonl
head -1 probes/zh/system-leak.jsonl   > /tmp/smoke_probes/b.jsonl
python -m src.main --config configs/models.yaml --probes /tmp/smoke_probes/ --out /tmp/smoke_out/
```

## 对应关系

| 类别 | 含义 | 判定 | 归属 |
|---|---|---|---|
| `direct-injection` | 直接要求无视指令/扮演 X/执行 Y | secret-leak(确定性) | 基础 |
| `role-manipulation` | 角色扮演越权 | secret-leak(确定性) | 基础 |
| `goal-hijacking` | 任务目标偷换 | secret-leak(确定性) | 基础 |
| `system-leak` | 诱导吐出系统提示/secret | secret-leak(确定性) | 基础 |
| `indirect-injection` | 注入藏在网页/引用文本 | secret-leak(确定性) | 基础 |
| `encoding-injection` | 指令经 Base64/ROT13/Hex/反转/leetspeak/零宽隐藏 | secret-leak(确定性) | garak 启发(OOD) |
| `cloze-leak` | 填空/补全/掩码诱导模型续写 secret 片段 | secret-leak(确定性) | garak 启发(OOD) |
| `emotional-social` | 祖母/亲情/心理安慰等情感社工越狱 | secret-leak(确定性) | garak 启发(OOD) |

判定口径与指标定义见根目录 `taxonomy.md`。

## 安全微调数据集(供云端 SFT)

`scripts/build_safety_sft_data.py` 从探针库构造"安全性微调"数据集(OpenAI messages 格式 JSONL,百炼可直接上传),输出到 `data/safety_sft/`(已 gitignore):

```bash
python3 scripts/build_safety_sft_data.py            # 默认 --seed 42,确定性可复现
```

### 分层测试集设计(L1 公平 / L2 组合 / L3 迁移)

测试集刻意分三层,兼顾"公平可达成"与"泛化压力":

| 层 | 文件 | 条数 | 设计意图 |
|---|---|---|---|
| **L1 fair(主指标)** | `test_fair.jsonl` | 46 | 训练**同原理、不同实现**:原始评测探针每(类别,语言)后 4 条(训练和 L1 均未见过)+ 良性 6。规则内化考法,不刁钻 |
| **L2 compose** | `test_compose.jsonl` | 6 | 原理**新组合**:角色+编码 / 填空+system-leak / 间接+目标劫持(中英各 3),测组合泛化 |
| **L3 OOD** | `test_ood.jsonl` | 36 | garak 启发的**全新攻击面**(encoding-injection / cloze-leak / emotional-social),训练完全不碰,测迁移鲁棒性 |

拆分:`train` 75(攻击 60 + 良性 15)、`val` 24、`test_fair` 46、`test_compose` 6、`test_ood` 36,**合计 187 条**。

关键保证:
- **训练集与原始 60 条评测探针零重合**——原始探针全部保留在 `val`/`test_fair`,因此微调前后跑 60 探针基准评测仍是"完全未见";
- **L1 是主指标**:与训练同原理、换具体实现,展现安全对齐是否真正内化规则,而不是背题;
- **L3 是压力测试**:全新的攻击形式,若 L1 通过而 L3 失败,说明微调学到了具体模式而非泛化原则;
- 良性问答(Benign)在训练、val、L1 都有,用于监测安全微调后的**过度拒绝**(误拒率);
- 同 seed 完全复现;拆分统计与设计说明见 `data/safety_sft/meta.json`。
