# engine/capabilities/pipeline.py
from __future__ import annotations
from typing import Dict, Any

# אימוץ יכולת חדשה תחת מדיניות: אם policy["capabilities"].get("manual_approve") True
# נחזיר {ok:False, action:{approve_capability:kind}}; אחרת נאמץ אוטומטית.

def capability_adoption_flow(kind: str, *, proposal: Dict[str,Any], policy: Dict[str,Any]) -> Dict[str,Any]:
    caps = (policy or {}).get("capabilities", {})
    if bool(caps.get("manual_approve", False)) and not proposal.get("approved"):
        return {"ok": False, "action": {"approve_capability": kind, "proposal": proposal}}
    # auto‑adopt
    return {"ok": True, "kind": kind, "adopted": True, "proposal": proposal}