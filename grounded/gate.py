# imu_repo/grounded/gate.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import time
from grounded.provenance_store import verify

class GateDenied(Exception):
    def __init__(self, reasons: List[str]): 
        super().__init__(";".join(reasons))
        self.reasons = reasons

def enforce_all(claims: List[Dict[str, Any]],
                *, 
                require_hmac: bool = True,
                min_trust: float = 0.7,
                max_age_s: int | None = None) -> Dict[str, Any]:
    """
    בודק שכל claim מסופק עם digest ראיה תקפה, עומד ב-Trust/TTL/חתימה.
    claim = {"digest": "...", "min_trust"?: float}
    """
    out = {"ok": False, "checked": [], "reasons": []}
    now = int(time.time())
    if not claims:
        out["reasons"].append("no_claims_provided")
        return out

    for c in claims:
        dg = c.get("digest")
        if not dg:
            out["reasons"].append("claim_missing_digest"); continue
        thr = float(c.get("min_trust", min_trust))
        v = verify(dg, require_hmac=require_hmac, min_trust=thr)
        if not v.get("ok"):
            out["reasons"].append(f"verify_failed:{dg}:{','.join(v.get('reasons',[]))}")
            continue
        meta = v.get("meta", {})
        if max_age_s is not None:
            ts = int(meta.get("fetched_at", 0))
            if ts and now - ts > max_age_s:
                out["reasons"].append(f"stale:{dg}"); 
                continue
        out["checked"].append({"digest": dg, "meta": meta})

    out["ok"] = len(out["checked"]) == len(claims) and len(claims) > 0 and len(out["reasons"]) == 0
    return out

def require(claims: List[Dict[str, Any]], **kw) -> List[Dict[str, Any]]:
    res = enforce_all(claims, **kw)
    if not res["ok"]:
        raise GateDenied(res["reasons"])
    return res["checked"]