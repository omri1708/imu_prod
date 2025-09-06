# engine/grounding_gate.py
from typing import Dict, Any, List
from ..policy.policy_enforcer import PolicyEnforcer, PolicyViolation
from ..policy.user_policy import UserPolicy

class GroundingError(Exception): ...

def require_grounded_response(policy: UserPolicy,
                              enforcer: PolicyEnforcer,
                              claims: List[Dict[str, Any]],
                              evidence_records: List[Dict[str, Any]]):
    try:
        enforcer.enforce_grounding(policy, claims, evidence_records)
    except PolicyViolation as e:
        raise GroundingError(str(e))