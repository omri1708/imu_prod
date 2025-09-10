from __future__ import annotations
import os, json
from typing import Any, Dict

STATE_F = "var/llm_bandit/state.json"
os.makedirs(os.path.dirname(STATE_F), exist_ok=True)

def load_state(default):
    try: return json.loads(open(STATE_F,"r",encoding="utf-8").read())
    except Exception: return default

def save_state(d: Dict[str,Any]):
    with open(STATE_F,"w",encoding="utf-8") as f: json.dump(d,f,ensure_ascii=False,indent=2)
