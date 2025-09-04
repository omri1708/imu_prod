# imu_repo/engine/capability_wrap.py
from __future__ import annotations
from typing import Callable, Awaitable, Any, Dict, Optional

from engine.pipeline_defaults import build_user_guarded
from grounded.claims import current
from engine.config import load_config
from engine.guard_enforce import enforce_guard_before_respond
from engine.errors import GuardRejection
from engine.fallbacks import safe_text_fallback
from engine.official_gate import run_official_checks 

TextCap = Callable[[Any], Awaitable[str]] | Callable[[Any], str]
GuardedTextCap = Callable[[Any], Awaitable[Dict[str,Any]]]


async def guard_text_capability_for_user(func: Callable[[Dict[str,Any]], Awaitable[str]], *, user_id: str):
    async def _wrapped(payload: Dict[str,Any]) -> Dict[str,Any]:
        cfg = load_config()
        try:
            text = await func(payload)
            # קודם מפעילים אימות רשמי (יאסוף official_verified אם אפשר)
            run_official_checks(cfg)
            # ואז אוכפים Guard כללי (שיכול לכלול דרישה ל-official_verified)
            enforce_guard_before_respond(evidences=current().snapshot(), cfg=cfg)
            return {"text": text, "claims": current().snapshot()}
        except GuardRejection as gr:
            fb = safe_text_fallback(reason=gr.reason, details=gr.details)
            return {"text": fb, "claims": current().snapshot(), "guard_rejected": True}
    return _wrapped


class CapabilityRegistry:
    """
    רישום ועטיפה פר-משתמש לכל יכולות טקסטואליות.
    """
    def __init__(self) -> None:
        self._caps: Dict[str, TextCap] = {}

    def register(self, name: str, fn: TextCap) -> None:
        if name in self._caps:
            raise KeyError(f"capability '{name}' already registered")
        self._caps[name] = fn

    def get(self, name: str) -> TextCap:
        return self._caps[name]

    async def guarded(self, name: str, *, user_id: Optional[str]) -> GuardedTextCap:
        cap = self.get(name)
        return await guard_text_capability_for_user(cap, user_id=user_id)
        
   # רישום גלובלי קטן
registry = CapabilityRegistry()