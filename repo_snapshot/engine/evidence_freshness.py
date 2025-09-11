# imu_repo/engine/evidence_freshness.py
from __future__ import annotations
import time
from typing import Dict, Any, List, Optional

class FreshnessError(Exception): ...

def _ev_ts(ev: Dict[str,Any]) -> Optional[float]:
    """
    מחלץ חותמת־זמן מן הראיה:
    עדיפות: ev["ts"] (epoch-seconds float)
    אם אין — מנסה ev["http_date_epoch"] (כבר מנותח), ואם אין — נכשל.
    """
    if "ts" in ev:
        try:
            return float(ev["ts"])
        except Exception:
            return None
    if "http_date_epoch" in ev:
        try:
            return float(ev["http_date_epoch"])
        except Exception:
            return None
    return None

def _claim_type(claim: Dict[str,Any]) -> str:
    t = str(claim.get("type") or "").strip().lower()
    return t or "generic"

def enforce_claims_freshness(claims: List[Dict[str,Any]], policy: Dict[str,Any], *, now: Optional[float]=None) -> None:
    """
    מדיניות:
      policy["freshness_sla_sec_by_type"] = {"latency": 600, "kpi": 900, ...}
      policy["default_freshness_sec"] = 3600  # אם אין התאמה לפי סוג
    כל claim חייב לפחות ראיה אחת טרייה מן ה-SLA.
    """
    now = time.time() if now is None else float(now)
    mapping = {str(k).lower(): float(v) for k,v in (policy.get("freshness_sla_sec_by_type") or {}).items()}
    default_sla = float(policy.get("default_freshness_sec", 0.0))  # 0 → לא אוכף ברירת־מחדל

    for c in (claims or []):
        ctype = _claim_type(c)
        sla = mapping.get(ctype, default_sla)
        if sla <= 0:
            # לא הוגדר SLA — דולג.
            continue
        evs = c.get("evidence") or []
        if not isinstance(evs, list) or not evs:
            raise FreshnessError(f"claim '{c.get('id','?')}' type={ctype} lacks evidence for freshness SLA {sla}s")
        ok = False
        oldest = None
        for ev in evs:
            ts = _ev_ts(ev)
            if ts is None:
                continue
            age = now - ts
            oldest = min(oldest, age) if oldest is not None else age
            if age <= sla:
                ok = True
                break
        if not ok:
            raise FreshnessError(f"claim '{c.get('id','?')}' type={ctype} evidence stale (min age={int(oldest or -1)}s > sla={int(sla)}s)")