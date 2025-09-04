# imu_repo/user_model/conflict_resolution.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import time

def resolve_preferences(candidates: List[Dict[str,Any]], *, now: float | None=None) -> Dict[str,Any]:
    """
    מכניס רשומות העדפה בסגנון:
      {"key":"theme","value":"dark","confidence":0.7,"added_at":...,"tier":"T1|T2"}
    ובוחר תוצאה לפי משקל=confidence * recency * tier_weight
    policy:
      T2 weight=1.2, T1 weight=1.0; recency = 1/(1+age_days)
      אם דחוס (< 1.3x) — מחזיר ask_user=True וגם שתי מועמדות מובילות.
    """
    if not candidates: return {"decided": False, "reason":"no_candidates"}
    now = time.time() if now is None else float(now)
    scored=[]
    for c in candidates:
        conf = float(c.get("confidence",0.5))
        age_days = max(0.0, (now - float(c.get("added_at",now)))/86400.0)
        rec = 1.0/(1.0+age_days)
        tier_w = 1.2 if c.get("tier")=="T2" else 1.0
        s = conf * rec * tier_w
        scored.append((s,c))
    scored.sort(key=lambda x:x[0], reverse=True)
    best, second = scored[0], (scored[1] if len(scored)>1 else None)
    decide = True
    ask = False
    if second:
        ratio = best[0] / (second[0]+1e-9)
        if ratio < 1.3:
            decide=False; ask=True
    out = {"decided": decide, "ask_user": ask, "winner": best[1]}
    if ask and second: out["runner_up"] = second[1]
    return out