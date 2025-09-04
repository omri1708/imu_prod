# imu_repo/grace/grace_manager.py
from __future__ import annotations
from typing import Dict, Any
import os, json, time

GRACE_PATH = "/mnt/data/imu_repo/history/grace_tokens.json"

def _load() -> Dict[str,Any]:
    if not os.path.exists(GRACE_PATH): return {"by_user":{}, "default_tokens":2}
    with open(GRACE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _save(st: Dict[str,Any]) -> None:
    os.makedirs(os.path.dirname(GRACE_PATH), exist_ok=True)
    with open(GRACE_PATH, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)

def grant(user_id: str, *, reason: str, ttl_s: int = 3600) -> Dict[str,Any]:
    st = _load()
    by = st["by_user"].setdefault(user_id, {"tokens": st.get("default_tokens",2), "graces":[]})
    if by["tokens"] <= 0:
        return {"ok": False, "reason":"no_tokens"}
    by["tokens"] -= 1
    g = {"reason": reason, "expire_ts": time.time() + ttl_s}
    by["graces"].append(g)
    _save(st)
    return {"ok": True, "grace": g, "remaining": by["tokens"]}

def active(user_id: str) -> bool:
    st = _load()
    by = st["by_user"].get(user_id)
    if not by: return False
    now = time.time()
    # ניקוי פגים
    by["graces"] = [g for g in by["graces"] if g["expire_ts"] > now]
    _save(st)
    return bool(by["graces"])

def refill(user_id: str, tokens: int = 2) -> None:
    st = _load()
    st["by_user"][user_id] = {"tokens": tokens, "graces":[]}
    _save(st)