# imu_repo/user_model/profile_store.py
from __future__ import annotations
import time
from typing import Dict, Any
from privacy.storage import save_json_encrypted, load_json_encrypted

def _load(user_id: str) -> Dict[str, Any]:
    return load_json_encrypted(user_id, "profile") or {"prefs": {}, "beliefs": {}, "affect": {}, "contradictions": []}

def _save(user_id: str, prof: Dict[str,Any]) -> None:
    save_json_encrypted(user_id, "profile", prof, ttl_s=None)

def get_profile(user_id: str) -> Dict[str, Any]:
    return _load(user_id)

def set_pref(user_id: str, key: str, value: Any, *, confidence: float = 0.8) -> Dict[str, Any]:
    prof = _load(user_id)
    prefs = prof.setdefault("prefs", {})
    ts = time.time()
    if key in prefs and prefs[key].get("value") != value:
        # סתירה — נשמר יומן
        prof.setdefault("contradictions", []).append({
            "key": key, "old": prefs[key], "new": {"value": value, "confidence": confidence, "ts": ts}, "ts": ts
        })
        # כלל הכרעה: בברירת מחדל — “החדש עם confidence גבוה יותר, או המאוחר”
        old_c = float(prefs[key].get("confidence", 0.0))
        if confidence >= old_c:
            prefs[key] = {"value": value, "confidence": confidence, "ts": ts}
    else:
        prefs[key] = {"value": value, "confidence": confidence, "ts": ts}
    _save(user_id, prof)
    return prof

def consolidate(user_id: str) -> Dict[str, Any]:
    """מאחד העדפות לפי כלל פשוט (חדש/בטוח גובר) — נשאר פשוט כדי לשמור על דטרמיניזם."""
    prof = _load(user_id)
    # כרגע אין מיזוג מורכב נוסף — כבר טופל בזמן set_pref
    _save(user_id, prof)
    return prof