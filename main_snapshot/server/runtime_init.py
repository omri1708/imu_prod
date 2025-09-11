from __future__ import annotations
import os

def ensure_runtime_dirs():
    for d in ("var/audit/pipeline", "var/llm_cache", "var/snapshots", "var/cas", "var/registry"):
        os.makedirs(d, exist_ok=True)
    # probe write
    with open("var/audit/pipeline/.probe","w") as f:
        f.write("ok")