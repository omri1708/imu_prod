# imu_repo/engine/provenance_gate.py
from __future__ import annotations
from typing import List, Dict, Any
from provenance.provenance import aggregate_trust, evidence_expired
from policy.policy_engine import PolicyEngine

class GateFailure(Exception): ...

def _count_sources(evs: List[Dict[str,Any]]) -> int:
    return len({e.get("source_url","") for e in evs if e.get("source_url")})

def enforce_evidence_gate(
    evs: List[Dict[str,Any]],
    *,
    domain: str | None = None,
    risk_hint: str | None = None,
    policy_engine: PolicyEngine | None = None
) -> Dict[str,Any]:
    pe = policy_engine or PolicyEngine()
    pol = pe.resolve(domain, risk_hint)  # {min_trust, max_ttl_s, min_sources, freshness_decay}
    if not evs:
        raise GateFailure("no evidences present")
    # הסר פגות־תוקף לפי max_ttl_s של המדיניות
    fresh = []
    from provenance.provenance import now_ts
    for e in evs:
        ts = int(e.get("ts", now_ts()))
        ttl = int(e.get("ttl_s", 3600))
        if now_ts() - ts > min(ttl, pol["max_ttl_s"]):
            continue
        fresh.append(e)
    if not fresh:
        raise GateFailure("all evidences expired by policy")
    agg = aggregate_trust(fresh)
    if agg < pol["min_trust"]:
        raise GateFailure(f"agg_trust {agg:.2f} < min_trust {pol['min_trust']:.2f}")
    if _count_sources(fresh) < pol["min_sources"]:
        raise GateFailure(f"not enough distinct sources (need {pol['min_sources']})")
    return {"agg_trust": agg, "count": len(fresh), "policy": pol}