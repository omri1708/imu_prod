# imu_repo/engine/provenance_gate.py
from __future__ import annotations
from typing import List, Dict, Any
from provenance.provenance import aggregate_trust, evidence_expired

class GateFailure(Exception): ...

def enforce_evidence_gate(evs: List[Dict[str,Any]], *, min_trust: float=0.75) -> Dict[str,Any]:
    if not evs:
        raise GateFailure("no evidences present")
    fresh = [e for e in evs if not evidence_expired(e)]
    if not fresh:
        raise GateFailure("all evidences expired")
    agg = aggregate_trust(fresh)
    if agg < min_trust:
        raise GateFailure(f"agg_trust {agg:.2f} < min_trust {min_trust:.2f}")
    return {"agg_trust": agg, "count": len(fresh)}