# engine/grounding_gate.py
from typing import Dict, Any, List
from ..policy.policy_enforcer import PolicyEnforcer, PolicyViolation
from ..policy.user_policy import UserPolicy
from typing import List, Dict, Any
from policy.model import UserPolicy
from provenance.sign import require_signed

class GroundingError(Exception): ...

def require_grounded_response(policy: UserPolicy,
                              enforcer: PolicyEnforcer,
                              claims: List[Dict[str, Any]],
                              evidence_records: List[Dict[str, Any]]):
    try:
        enforcer.enforce_grounding(policy, claims, evidence_records)
    except PolicyViolation as e:
        raise GroundingError(str(e))
    
TRUST_LEVELS = {"low":0,"medium":1,"high":2}

def _enough_trust(source_level: str, min_level: str) -> bool:
    return TRUST_LEVELS.get(source_level,0) >= TRUST_LEVELS.get(min_level,1)

def enforce_grounding(policy: UserPolicy, claims: List[Dict[str,Any]]) -> None:
    if not policy.strict_grounding:
        return
    if not claims:
        raise ValueError("no claims present – grounding_required")
    # לכל claim נדרשת ראיה חתומה ו־trustlevel מספק
    for c in claims:
        ev = c.get("evidence") or []
        if not ev:
            raise ValueError("claim missing evidence")
        ok_any = False
        for e in ev:
            # חתימה/עטיפה
            if "digest" in e:
                require_signed(e["digest"])  # יזרוק אם לא חתום/לא קיים
            level = e.get("trust","low")
            if _enough_trust(level, policy.min_source_trust):
                ok_any = True
        if not ok_any:
            raise ValueError("no evidence met min trust")