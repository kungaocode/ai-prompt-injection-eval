"""汇总层:ASR / FPR / 一致性,渲染成 markdown 报告。"""

from __future__ import annotations

from .schemas import JudgeResult


def compute_asr(results: list[JudgeResult]) -> dict:
    """返回 {overall: (成功数, 总数), by_category: {类别: (成功数, 总数)}}。"""
    total_s = total_t = 0
    cats: dict[str, tuple[int, int]] = {}
    for r in results:
        total_t += 1
        total_s += int(r.success)
        s, t = cats.get(r.category, (0, 0))
        cats[r.category] = (s + int(r.success), t + 1)
    return {"overall": (total_s, total_t),
            "by_category": dict(sorted(cats.items()))}


def pct(s: int, t: int) -> str:
    return f"{100.0 * s / t:.1f}%" if t else "n/a"


def render(cfg: dict, per_model: dict, fpr: float | None = None,
           agreement: float | None = None) -> str:
    """per_model: {模型名: {"baseline": asr, "guard": asr|None}}。"""
    secret = cfg["secret"]
    lines = [
        "# Prompt-Injection Evaluation Report",
        "",
        f"- secret: `{secret}`",
        f"- targets (cloud API): {', '.join(per_model.keys())}",
        "",
        "## Attack Success Rate (ASR)",
        "",
        "| model | guard | category | ASR |",
        "|---|---|---|---|",
    ]
    for model, entry in per_model.items():
        base = entry["baseline"]
        for cat, (s, t) in base["by_category"].items():
            lines.append(f"| {model} | off | {cat} | {pct(s, t)} ({s}/{t}) |")
        s, t = base["overall"]
        lines.append(f"| {model} | off | **overall** | **{pct(s, t)} ({s}/{t})** |")
        grd = entry.get("guard")
        if grd:
            for cat, (s2, t2) in grd["by_category"].items():
                lines.append(f"| {model} | on | {cat} | {pct(s2, t2)} ({s2}/{t2}) |")
            s2, t2 = grd["overall"]
            lines.append(f"| {model} | on | **overall** | **{pct(s2, t2)} ({s2}/{t2})** |")
    lines.append("")

    if fpr is not None:
        lines += [
            "## Benign False-Positive Rate (FPR)",
            "",
            f"- guard input-filter blocks **{fpr:.1%}** of benign inputs",
            "",
        ]
    if agreement is not None:
        lines += [
            "## Judge–Human Agreement",
            "",
            f"- **{agreement:.1%}**",
            "",
        ]
    return "\n".join(lines)
