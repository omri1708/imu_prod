# imu_repo/user_model/consent.py
from __future__ import annotations
from typing import Dict, Any
import os, json, time
from user_model.identity import user_dir

CONSENT_FN = "consent.json"

def _path(user_key: str) -> str:
    return os.path.join(user_dir(user_key), CONSENT_FN)

def set_consent(user_key: str, purpose: str, *, granted: bool, ttl_s: int=365*24*3600, policy: str="v1") -> None:
    p = _path(user_key)
    try:
        data = json.load(open(p,"r",encoding="utf-8"))
    except Exception:
        data = {}
    data[purpose] = {"granted": bool(granted), "ts": time.time(), "ttl_s": int(ttl_s), "policy": policy}
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(data, open(p,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

def revoke(user_key: str, purpose: str) -> None:
    p = _path(user_key)
    try:
        data = json.load(open(p,"r",encoding="utf-8"))
    except Exception:
        data = {}
    if purpose in data:
        data[purpose]["granted"] = False
        data[purpose]["ts"] = time.time()
    json.dump(data, open(p,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

def check(user_key: str, purpose: str) -> Dict[str,Any]:
    p = _path(user_key)
    try:
        data = json.load(open(p,"r",encoding="utf-8"))
    except Exception:
        return {"ok": False, "reason": "no_record"}
    rec = data.get(purpose)
    if not rec: return {"ok": False, "reason": "no_record"}
    if not rec.get("granted", False): return {"ok": False, "reason": "revoked"}
    alive = (time.time() <= rec["ts"] + rec["ttl_s"])
    return {"ok": alive, "reason": None if alive else "expired", "record": rec}