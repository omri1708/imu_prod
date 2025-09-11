# imu_repo/engine/learn_store.py
from __future__ import annotations
import os, json, time, hashlib
from typing import Dict, Any, List, Tuple, Optional
from grounded.provenance import STORE as PROV_STORE

LEARN_DIR = os.path.join(PROV_STORE, "learn")
os.makedirs(LEARN_DIR, exist_ok=True)

def _task_key(name: str, goal: str) -> str:
    h = hashlib.sha256(goal.encode("utf-8")).hexdigest()[:16]
    return f"{name}__{h}"

def history_path(key: str) -> str:
    return os.path.join(LEARN_DIR, f"{key}.history.jsonl")

def baseline_path(key: str) -> str:
    return os.path.join(LEARN_DIR, f"{key}.baseline.json")

def append_history(key: str, entry: Dict[str,Any]) -> None:
    entry = dict(entry, ts=time.time())
    with open(history_path(key), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def load_history(key: str, limit: int = 500) -> List[Dict[str,Any]]:
    p = history_path(key)
    if not os.path.exists(p): return []
    out: List[Dict[str,Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out[-limit:]

def load_baseline(key: str) -> Optional[Dict[str,Any]]:
    p = baseline_path(key)
    if not os.path.exists(p): return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_baseline(key: str, data: Dict[str,Any]) -> None:
    tmp = baseline_path(key) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, baseline_path(key))