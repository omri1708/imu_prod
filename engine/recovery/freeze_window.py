# engine/recovery/freeze_window
from __future__ import annotations
import os, json, time
from typing import Dict, Any
from audit.merkle_log import MerkleAudit

BASE = "var/recovery"
STATE_F = os.path.join(BASE, "freeze_state.json")
AUDIT = MerkleAudit("var/audit/pipeline")
os.makedirs(BASE, exist_ok=True)

_DEF = {"windows": {}, "limits": {}}  # key -> {until, reason, ts} ; limits: daily counters

def _load() -> Dict[str,Any]:
    if not os.path.exists(STATE_F): return dict(_DEF)
    try: return json.loads(open(STATE_F,"r",encoding="utf-8").read())
    except Exception: return dict(_DEF)

def _save(d: Dict[str,Any]) -> None:
    with open(STATE_F,"w",encoding="utf-8") as f: json.dump(d,f,ensure_ascii=False,indent=2)

def start_freeze(key: str, *, minutes: int, reason: str, daily_cap: int = 4) -> Dict[str,Any]:
    now = time.time(); until = now + minutes*60
    d = _load()
    # מניעת רצף אין-סופי: cap יומי
    day = int(now//86400)
    cap_key = f"{key}:{day}"
    used = int((d.get("limits") or {}).get(cap_key, 0))
    if used >= daily_cap:
        AUDIT.append("freeze.skip_cap", {"key": key, "reason": reason, "day": day, "used": used})
        return {"ok": False, "cap_reached": True, "used": used, "daily_cap": daily_cap}
    d.setdefault("limits", {})[cap_key] = used + 1

    d.setdefault("windows",{})[key] = {"until":until, "reason":reason, "ts":now}
    _save(d)
    AUDIT.append("freeze.start", {"key": key, "until": until, "reason": reason})
    return {"ok": True, "key": key, "until": until}

def is_frozen(key: str) -> Dict[str,Any]:
    d = _load(); w = (d.get("windows") or {}).get(key)
    if not w: return {"ok": False, "frozen": False}
    if float(w.get("until",0)) < time.time():
        d["windows"].pop(key, None); _save(d)
        AUDIT.append("freeze.end", {"key": key})
        return {"ok": False, "frozen": False}
    return {"ok": True, "frozen": True, "until": w["until"], "reason": w.get("reason")}

