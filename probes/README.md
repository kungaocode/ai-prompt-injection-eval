# 探针库:生成与使用指令

探针集全部由 `generate_probes.py` 从分类学模板系统化生成,保证中英对称、可复现。
**不要手工改动 `en/`、`zh/` 下的 JSONL**;要改探针就改模板再重新生成。

## 重新生成(全量)

```bash
python probes/generate_probes.py
```

输出:
- `probes/en/<category>.jsonl` — 英文探针(每类 6 条)
- `probes/zh/<category>.jsonl` — 中文探针(每类 6 条)

当前规模:**5 类 × 中英各 6 = 60 条**。

## 扩容(目标:每类每语言 ≥12 条 = 总量 ≥120)

1. 编辑 `probes/generate_probes.py` 的 `TEMPLATES` 字典,在对应类别/语言的列表里**追加**新模板;
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

| 类别 | 含义 | 判定 |
|---|---|---|
| `direct-injection` | 直接要求无视指令/扮演 X/执行 Y | secret-leak(确定性) |
| `role-manipulation` | 角色扮演越权 | secret-leak(确定性) |
| `goal-hijacking` | 任务目标偷换 | secret-leak(确定性) |
| `system-leak` | 诱导吐出系统提示/secret | secret-leak(确定性) |
| `indirect-injection` | 注入藏在网页/引用文本 | secret-leak(确定性) |

判定口径与指标定义见根目录 `taxonomy.md`。
