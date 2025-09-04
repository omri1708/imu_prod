# imu_repo/engine/capability_wrap.py
from __future__ import annotations
from typing import Callable, Awaitable, Any, Dict, Optional

from engine.pipeline_defaults import build_user_guarded

TextCap = Callable[[Any], Awaitable[str]] | Callable[[Any], str]
GuardedTextCap = Callable[[Any], Awaitable[Dict[str,Any]]]

async def guard_text_capability_for_user(
    cap: TextCap, *, user_id: Optional[str]
) -> GuardedTextCap:
    """
    עוטף capability שמחזיר מחרוזת כך שיחזיר {"text":..., "claims":[...]}
    תחת Strict-Grounded per-user (כולל Evidences חובה, חתימה ו-Provenance).
    """
    wrapped = await build_user_guarded(cap, user_id=user_id)

    async def _wrapped(x: Any) -> Dict[str,Any]:
        out = await wrapped(x)
        assert isinstance(out, dict) and "text" in out and "claims" in out
        return out
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