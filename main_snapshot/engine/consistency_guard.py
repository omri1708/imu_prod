# imu_repo/engine/consistency_guard.py
from __future__ import annotations
import os, json, math, time
from typing import Dict, Any, List, Optional, Tuple

class ConsistencyError(Exception): ...

def _state_dir() -> str:
    d = os.environ.get("IMU_STATE_DIR") or ".state"
    os.makedirs(d, exist_ok=True)
    return d

def _state_path(name: str) -> str:
    return os.path.join(_state_dir(), name)

def _load_json(path: str) -> Dict[str,Any]:
    if not os.path.exists(path): return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _store_json(path: str, obj: Dict[str,Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _num(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None

def _default_cfg(policy: Dict[str,Any]) -> Dict[str,Any]:
    c = (policy or {}).get("consistency") or {}
    return {
        "drift_pct": float(c.get("drift_pct", 0.20)),  # 20% ברירת מחדל
        "near_miss_streak_heal_threshold": int(c.get("near_miss_streak_heal_threshold", 3)),
        "heal_action": str(c.get("heal_action", "freeze_autotune")),  # freeze_autotune | rollback | raise_require_fresh
        "rollback_factor": float(c.get("rollback_factor", 0.5))  # בעת rollback — להקטין אחוז פי 0.5
    }

def _group_key(claim: Dict[str,Any]) -> Optional[str]:
    return claim.get("consistency_group")

def check_drift_and_update(
    claims: List[Dict[str,Any]],
    *,
    policy: Dict[str,Any],
    stage_name: str,
    percent: int,
    near_miss: bool
) -> Dict[str,Any]:
    """
    משווה ערכי claims לקודמים לפי consistency_group.
    אם drift מעל סף → מדווח.
    מנהל מונה near_miss רציף לקבוצה, וממליץ Self-Heal.
    שומר מצב ב-.state/consistency.json
    """
    cfg = _default_cfg(policy)
    stpath = _state_path("consistency.json")
    st = _load_json(stpath)
    groups = st.get("groups") or {}  # { group: { "value": <last>, "ts": <epoch>, "near_miss_streak": int } }

    drifts = []
    updated = False
    now = time.time()

    for c in claims or []:
        g = _group_key(c)
        if not g: 
            continue
        val = _num(c.get("value"))
        if val is None:
            # לא ניתן למדוד drift — נדלג
            continue
        prev = groups.get(g)
        if prev is not None:
            prev_val = _num(prev.get("value"))
            if prev_val not in (None, 0):
                delta = abs(val - prev_val) / abs(prev_val)
                if delta > cfg["drift_pct"]:
                    drifts.append({"group": g, "prev": prev_val, "cur": val, "delta": delta})
        # עדכון ערך נוכחי
        groups[g] = {"value": val, "ts": now, "near_miss_streak": int(prev.get("near_miss_streak",0)) if prev else 0}
        updated = True

    # ניהול near_miss streak (אחיד לכל stage או פר קבוצה: נעדכן בכל קבוצה)
    if claims:
        for c in claims:
            g = _group_key(c)
            if not g: 
                continue
            rec = groups.get(g) or {"near_miss_streak": 0}
            if near_miss:
                rec["near_miss_streak"] = int(rec.get("near_miss_streak",0)) + 1
            else:
                rec["near_miss_streak"] = 0
            rec["ts"] = now
            groups[g] = rec
            updated = True

    st["groups"] = groups
    if updated:
        _store_json(stpath, st)

    # המלצת Self-Heal
    heal: Optional[Dict[str,Any]] = None
    trigger = False
    reason = None

    # תנאי הפעלה: או שיש drift, או שה־near_miss streak עבר סף
    if drifts:
        trigger = True
        reason = "drift"
    else:
        # אם יש לפחות קבוצה אחת שעברה סף streak
        for g, rec in groups.items():
            if int(rec.get("near_miss_streak",0)) >= int(cfg["near_miss_streak_heal_threshold"]):
                trigger = True
                reason = f"near_miss_streak({g})"
                break

    if trigger:
        action = cfg["heal_action"]
        if action == "rollback":
            heal = {"action":"rollback", "rollback_factor": cfg["rollback_factor"], "reason": reason}
        elif action == "raise_require_fresh":
            heal = {"action":"raise_require_fresh", "reason": reason}
        else:
            heal = {"action":"freeze_autotune", "reason": reason}

    return {
        "ok": True,
        "drifts": drifts,
        "heal": heal,
        "cfg": cfg
    }