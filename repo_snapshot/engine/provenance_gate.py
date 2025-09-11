# imu_repo/engine/provenance_gate.py
from __future__ import annotations
from typing import List, Dict, Any
from provenance.provenance import aggregate_trust, now_ts
from policy.freshness_profile import get_profile
from policy.policy_rules import PolicyEngine
import os, json

class GateFailure(Exception): ...

def _count_sources(evs: List[Dict[str,Any]]) -> int:
    return len({e.get("source_url","") for e in evs if e.get("source_url")})

def _load_policy_from_disk() -> dict | None:
    p = os.environ.get("IMU_POLICY_PATH", "/mnt/data/.imu_policy.json")
    if os.path.exists(p):
        try:
            with open(p,"r",encoding="utf-8") as f: return json.load(f)
        except Exception:
            return None
    return None

def enforce_evidence_gate(
    evs: List[Dict[str,Any]],
    *,
    domain: str | None = None,
    risk_hint: str | None = None,
    policy_engine: PolicyEngine | None = None
) -> Dict[str,Any]:
    pe = policy_engine or PolicyEngine(_load_policy_from_disk() or None)
    pol = pe.resolve(domain, risk_hint)  # {min_trust, max_ttl_s, min_sources, ...}
    if not evs:
        raise GateFailure("no evidences present")

    fresh = []
    for e in evs:
        prof = get_profile(e.get("kind"))
        # השתמש ב־min בין max_ttl_s של הפרופיל לבין מדיניות כללית
        ttl_cap = min(int(e.get("ttl_s", 3600)), int(prof["max_ttl_s"]), int(pol["max_ttl_s"]))
        ts = int(e.get("ts", now_ts()))
        if now_ts() - ts > ttl_cap:
            continue
        e2 = dict(e)
        e2["ttl_s"] = ttl_cap
        fresh.append(e2)

    if not fresh:
        raise GateFailure("all evidences expired by policy/profile")

    agg = aggregate_trust(fresh)
    if agg < pol["min_trust"]:
        raise GateFailure(f"agg_trust {agg:.2f} < min_trust {pol['min_trust']:.2f}")
    if _count_sources(fresh) < pol["min_sources"]:
        raise GateFailure(f"not enough distinct sources (need {pol['min_sources']})")
    return {"agg_trust": agg, "count": len(fresh), "policy": pol}