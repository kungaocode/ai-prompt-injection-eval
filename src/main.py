"""入口:跑探针 → 判定 → 汇总,一条命令出报告。

用法示例:
    python -m src.main --config configs/models.yaml --probes probes/ --out results/      # 真实云 API
    python -m src.main --mock --probes probes/ --out results/                            # 离线跑通
    python -m src.main --mock --guard --benign benign/benign_inputs.jsonl --out results/ # 带 guard + FPR
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from .guard import Guard
from .judge import judge_record
from .report import compute_asr, render
from .runner import (build_client, chat, load_config, load_probes,
                     mock_output, system_prompt)
from .schemas import Record


def run_probes(cfg: dict, probes, model: str, guard: Guard | None,
               client, mock: bool) -> list[Record]:
    sys = system_prompt(cfg)
    records: list[Record] = []
    for probe in probes:
        gr = guard.apply(sys, probe.text) if guard else None
        if gr and gr.blocked:
            output = "[BLOCKED by input-filter]"
        elif mock:
            output = mock_output(probe.category, cfg["secret"])
        else:
            system = gr.system if gr else sys
            user = gr.user if gr else probe.text
            output = chat(client, model, system, user)
        # 输出层校验(若启用):泄露即拦截
        if guard and guard.output_check and guard.check_output(output)[0]:
            output = "[BLOCKED by output-check]"
        records.append(Record(probe_id=probe.id, model=model,
                              guard=guard is not None, output=output))
    return records


def judge_records(probes_by_id, records, judge_client, judge_model):
    return [judge_record(judge_client, judge_model, probes_by_id[r.probe_id], r)
            for r in records]


def main() -> None:
    p = argparse.ArgumentParser(description="自研 prompt-injection 评测")
    p.add_argument("--config", default="configs/models.yaml")
    p.add_argument("--probes", default="probes/")
    p.add_argument("--out", default="results/")
    p.add_argument("--benign", default=None, help="良性输入集(算 FPR)")
    p.add_argument("--mock", action="store_true", help="离线假模型跑通流水线")
    p.add_argument("--guard", action="store_true", help="加 guard 做 2×2 复测")
    args = p.parse_args()

    cfg = load_config(args.config)
    probes = load_probes(args.probes)
    probes_by_id = {p.id: p for p in probes}

    if args.mock:
        client = judge_client = None
    else:
        env_key = cfg["dashscope"]["api_key_env"]
        if not os.environ.get(env_key):
            print(f"[!] 缺少环境变量 {env_key},请在 .env 中设置(见 .env.example),"
                  f"或用 --mock 离线跑通。")
            raise SystemExit(1)
        client = build_client(cfg)
        judge_client = client  # judge 用同一云服务上的不同模型名

    judge_model = cfg["default_judge"]
    guard = Guard(secret=cfg["secret"]) if args.guard else None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    all_records, all_results = [], []
    per_model: dict = {}

    for target in cfg["targets"]:
        model = target["name"]
        for use_guard in ([False, True] if guard else [False]):
            g = guard if use_guard else None
            records = run_probes(cfg, probes, model, g, client, args.mock)
            results = judge_records(probes_by_id, records, judge_client, judge_model)
            all_records += records
            all_results += results
            per_model.setdefault(model, {})[
                "guard" if use_guard else "baseline"] = compute_asr(results)

    # 保存原始记录与判定(可复现)
    (out / "records.jsonl").write_text(
        "\n".join(json.dumps(asdict(r), ensure_ascii=False) for r in all_records)
        + "\n", encoding="utf-8")
    (out / "judge_results.jsonl").write_text(
        "\n".join(json.dumps(asdict(r), ensure_ascii=False) for r in all_results)
        + "\n", encoding="utf-8")

    # 良性 FPR
    fpr = None
    if args.benign and guard:
        blocked = total = 0
        for line in Path(args.benign).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            total += 1
            b, _ = guard.filter_input(d["text"])
            blocked += int(b)
        fpr = blocked / total if total else None

    md = render(cfg, per_model, fpr)
    (out / "report.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
