# policy/auto_approval.py
# RBAC-aware auto approval: decide if we can auto-approve gates result.
from __future__ import annotations
from typing import Dict, Any
from policy.rbac import RBAC_DB

def auto_approve(gates: Dict[str,Any], user_id: str) -> Dict[str,Any]:
    """
    gates: {"ok": bool, "reasons": [...]} as from Gatekeeper /evaluate
    Logic:
      - admin: approve unless explicit "checks:failed" or "evidence:*:insufficient"
      - dev: approve if ok==True; otherwise deny.
      - viewer: never approve.
    """
    perms = RBAC_DB.list_user_perms(user_id)
    roles = RBAC_DB.users.get(user_id).roles if user_id in RBAC_DB.users else []
    reasons = gates.get("reasons") or []
    if "admin" in roles or any(p=="*" for p in perms):
        deny_markers = [r for r in reasons if r.startswith("evidence:") or r.startswith("checks:failed")]
        if not deny_markers:
            return {"approve": True, "reason": "admin override"}
        return {"approve": False, "reason": f"admin sees hard deny: {deny_markers}"}
    if "dev" in roles:
        return {"approve": bool(gates.get("ok")), "reason": "dev follows gates"}
    return {"approve": False, "reason": "viewer cannot auto-approve"}