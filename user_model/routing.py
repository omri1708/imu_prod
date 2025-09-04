from __future__ import annotations
import os, json
from typing import Dict, Any, Optional, List
from user_model.memory import UserMemory

ROOT = "/mnt/data/imu_repo"
STATE_DIR = os.path.join(ROOT, "state")
os.makedirs(STATE_DIR, exist_ok=True)
USERS_FILE = os.path.join(STATE_DIR, "users.json")

_DEFAULT = {
    "min_trust": 0.7,
    "max_age_s": 3600,
    "strict_grounded": True
}

def _load() -> Dict[str,Any]:
    if not os.path.exists(USERS_FILE):
        return {"users": {}}
    try:
        return json.load(open(USERS_FILE, "r", encoding="utf-8"))
    except Exception:
        return {"users": {}}

def _save(obj: Dict[str,Any]) -> None:
    tmp = USERS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2))
    os.replace(tmp, USERS_FILE)

def get_profile(user_id: str) -> Dict[str,Any]:
    db = _load()
    return db.get("users", {}).get(user_id, dict(_DEFAULT))

def set_profile(user_id: str, **kwargs: Any) -> Dict[str,Any]:
    db = _load()
    u = db.get("users", {}).get(user_id, dict(_DEFAULT))
    for k,v in kwargs.items():
        u[k] = v
    db.setdefault("users", {})[user_id] = u
    _save(db)
    return u

def resolve_gate(user_id: Optional[str]) -> Dict[str,Any]:
    prof = dict(_DEFAULT)
    if user_id:
        prof.update(get_profile(user_id))
    # החזר ספי Gate לשימוש במעטפות
    return {
        "min_trust": float(prof.get("min_trust", _DEFAULT["min_trust"])),
        "max_age_s": int(prof.get("max_age_s", _DEFAULT["max_age_s"])),
        "strict_grounded": bool(prof.get("strict_grounded", True)),
    }


def reorder_lang_pref(user_id: str, lang_pref: List[str]) -> List[str]:
    if not lang_pref: return []
    mem = UserMemory()
    prof = mem.read_profile(user_id)
    pref = (prof.get("prefs") or {}).get("lang_pref") or {}
    v = pref.get("value")
    if v and v in lang_pref:
        return [v] + [x for x in lang_pref if x != v]
    return lang_pref