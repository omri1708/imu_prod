# imu_repo/engine/agent_emit.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from engine.respond_bridge import respond_with_required_proof

def agent_emit_answer(
    *,
    answer_text: str,
    ctx: Optional[Dict[str,Any]] = None,
    claims: Optional[List[Dict[str,Any]]] = None,
    policy: Optional[Dict[str,Any]] = None
) -> Dict[str,Any]:
    """
    זוהי עטיפה לשימוש ישיר ע"י סוכן/פייפליין:
      - אפשר להעביר claims מפורשים (למשל מה-Planner/Verifier)
      - או לסמוך על ctx["__claims__"] שהורכבו קודם לכן ב-Vetting.
    """
    return respond_with_required_proof(
        response_text=answer_text,
        ctx=ctx,
        extra_claims=claims,
        policy=policy
    )