# imu_repo/user_model/conflict_resolution.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import time
import math

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