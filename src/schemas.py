"""评测流水线共享的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Probe:
    """一条攻击探针。"""

    id: str
    category: str        # direct-injection / role-manipulation / goal-hijacking / system-leak / indirect-injection
    lang: str            # en / zh
    text: str            # 用户消息正文(攻击载荷)
    criterion: str = "secret-leak"   # 判定口径,当前 MVP 统一用 secret-leak(确定性)
    secret: str | None = None        # 确定性判定所用的 secret(高熵 token)

    @classmethod
    def from_dict(cls, d: dict) -> "Probe":
        return cls(**d)


@dataclass
class Record:
    """一次模型调用的原始记录。"""

    probe_id: str
    model: str
    guard: bool          # 是否加了 guard
    output: str          # 模型原始输出


@dataclass
class JudgeResult:
    """一条记录的判定结果。"""

    probe_id: str
    model: str
    guard: bool
    category: str
    judge_type: str      # deterministic / llm
    success: bool
    evidence: str | None = None
