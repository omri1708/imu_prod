# imu_repo/engine/guard_all.py
from __future__ import annotations
from typing import Callable, Awaitable, Any, Optional, Dict

from engine.pipeline_defaults import build_user_guarded

async def guard_text_handler_for_user(
    handler: Callable[[Any], Awaitable[str]] | Callable[[Any], str],
    *,
    user_id: Optional[str]
) -> Callable[[Any], Awaitable[Dict[str,Any]]]:
    """
    עוטף handler שמחזיר מחרוזת כך שיחזיר {"text": ..., "claims":[...]}
    תחת Strict-Grounded per-user (כולל Evidences חובה, חתימה ו-Provenance).
    """
    wrapped = await build_user_guarded(handler, user_id=user_id)

    async def _wrapped(x: Any) -> Dict[str,Any]:
        out = await wrapped(x)   # already {"text": ..., "claims":[...]}
        # הבטחת טיפוס
        assert isinstance(out, dict) and "text" in out and "claims" in out
        return out

    return _wrapped