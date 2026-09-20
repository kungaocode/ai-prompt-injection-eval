"""模型调用层:云 API(OpenAI 兼容,默认 DashScope)+ 离线 mock。

被测对象全部走云 API,免本地部署。支持 DashScope / 任意 OpenAI 兼容端点,
也保留 mock 模式用于无 key 时离线跑通流水线。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

from .schemas import Probe

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def load_config(path: str = "configs/models.yaml") -> dict:
    load_dotenv()  # 读取 .env(api key 等)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def system_prompt(cfg: dict) -> str:
    """把 secret 注入系统提示模板。"""
    return cfg["system_template"].format(secret=cfg["secret"])


def build_client(cfg: dict) -> OpenAI:
    """构造 OpenAI 兼容客户端(默认指向 DashScope 兼容模式端点)。"""
    dash = cfg.get("dashscope", {})
    base_url = dash.get("base_url", DASHSCOPE_BASE_URL)
    env_key = dash.get("api_key_env", "DASHSCOPE_API_KEY")
    return OpenAI(api_key=os.environ.get(env_key, ""), base_url=base_url)


def chat(client: OpenAI, model: str, system: str, user: str,
         temperature: float = 0.0) -> str:
    """一次对话补全,固定 temperature=0 保证可复现。"""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        extra_body={"enable_thinking": False},  # DashScope Qwen3 微调部署非流式必需
    )
    return resp.choices[0].message.content or ""


def load_probes(probe_dir: str) -> list[Probe]:
    """递归读取 probes/ 下所有 *.jsonl。"""
    probes: list[Probe] = []
    for jl in sorted(Path(probe_dir).rglob("*.jsonl")):
        for line in jl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                probes.append(Probe.from_dict(json.loads(line)))
    return probes


# ---- 离线 mock:模拟一个"对 direct-injection / system-leak 脆弱、其余免疫"的假模型 ----
# 仅用于无 API key 时验证流水线;真实数字必须跑真实云 API。
MOCK_VULNERABLE = {"direct-injection", "system-leak"}


def mock_output(category: str, secret: str) -> str:
    if category in MOCK_VULNERABLE:
        return f"(mock) The system secret is {secret}."
    return "(mock) I'm sorry, but I can't reveal that information."
