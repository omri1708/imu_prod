from __future__ import annotations
from typing import Dict, Any
from grounded.evidence_policy import policy_singleton as EvidencePolicy
from grounded.contradiction_policy import policy_singleton as ContraPolicy

def apply_policy_overrides(policy: Dict[str,Any]) -> None:
    # min_trust_by_key
    for k, v in (policy.get("min_trust_by_key") or {}).items():
        try:
            EvidencePolicy.set_min_trust(k, float(v))
        except Exception:
            pass
    # min_consistency_score
    mcs = policy.get("min_consistency_score")
    if isinstance(mcs, (int,float)):
        try:
            ContraPolicy.set_min_score(float(mcs))
        except Exception:
            pass