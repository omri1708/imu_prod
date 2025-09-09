# engine/prebuild/discovery.py
from __future__ import annotations
from typing import Any, Dict, List

LLM_FN = None  # ניתן להזריק פונקציית LLM חיצונית

def set_llm(fn):
    global LLM_FN; LLM_FN = fn

COMMON_HINTS = [
    ("android", {"name":"android-sdk", "pkg":"android-sdk", "manager":"brew", "spdx":"Apache-2.0"}),
    ("k8s",     {"name":"kubectl", "pkg":"kubectl", "manager":"brew", "spdx":"Apache-2.0"}),
    ("ffmpeg",  {"name":"ffmpeg", "pkg":"ffmpeg", "manager":"brew", "spdx":"GPL-3.0"}),
]

def infer_tools_required(spec: Any, ctx: Dict[str,Any]) -> List[Dict[str,Any]]:
    txt = str(spec)
    out: List[Dict[str,Any]] = []
    for kw, tool in COMMON_HINTS:
        if kw.lower() in txt.lower():
            out.append(tool)
    if LLM_FN:
        # הרחבה עם LLM: הצע כלים לפי תיאור
        try:
            rsp = LLM_FN({"task":"infer tools", "spec": spec, "ctx": ctx})
            out += rsp.get("tools", [])
        except Exception:
            pass
    return out