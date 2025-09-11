# imu_repo/user_model/conflict_resolution.py
from __future__ import annotations
from typing import List, Dict, Any
import time
import math

def resolve_preferences(candidates: List[Dict[str,Any]], *, now: float | None=None) -> Dict[str,Any]:
    """
    מכניס רשומות העדפה בסגנון:
      {"key":"theme","value":"dark","confidence":0.7,"added_at":...,"tier":"T1|T2"}
    ובוחר תוצאה לפי משקל=confidence * recency * tier_weight
    policy:
      T2 weight=1.2, T1 weight=1.0; recency = 1/(1+age_days)
      אם דחוס (< 1.3x) — מחזיר ask_user=True וגם שתי מועמדות מובילות.
    """
    if not candidates:
        return {"decided": False, "reason":"no_candidates"}
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
            decide=False
            ask=True
    out = {"decided": decide, "ask_user": ask, "winner": best[1]}
    if ask and second:
        out["runner_up"] = second[1]
    return out

def _w(rec: Dict[str,Any]) -> float:
    # משקל לפי trust * confidence * recency_factor
    trust = float(rec.get("trust", 0.5))
    conf  = float(rec.get("confidence", 0.5))
    ts    = float(rec.get("_ts", time.time()))
    age_s = max(1.0, time.time() - ts)
    recency = 1.0 / math.log1p(age_s)  # חדש → משקל גבוה
    return max(0.0, trust) * max(0.0, conf) * recency


def resolve_records(records: List[Dict[str,Any]], value_key: str="value") -> Dict[str,Any]:
    """
    records: [ {value: <...>, trust, confidence, _ts }, ... ]
    בוחר את הערך עם סכום משקלים הגבוה ביותר (ווטינג לפי ערכים).
    """
    buckets: Dict[str, float] = {}
    examples: Dict[str, List[Dict[str,Any]]] = {}
    for r in records:
        v = str(r.get(value_key))
        w = _w(r)
        buckets[v] = buckets.get(v, 0.0) + w
        examples.setdefault(v, []).append(r)
    if not buckets:
        return {"ok": False, "reason":"no_records"}
    chosen = max(buckets.items(), key=lambda kv: kv[1])[0]
    return {"ok": True, "chosen": chosen, "weights": buckets, "examples": examples}