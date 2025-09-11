# imu_repo/engine/capability_wrap.py
from __future__ import annotations
import time
from typing import Callable, Awaitable, Any, Dict, Optional

from engine.pipeline_defaults import build_user_guarded
from grounded.claims import current
from engine.config import load_config
from engine.guard_enforce import enforce_guard_before_respond
from engine.errors import GuardRejection, BudgetExceeded
from engine.fallbacks import safe_text_fallback
from engine.official_gate import run_official_checks 
from engine.async_sandbox import ASYNC_CAPS
from engine.phi_budget import consume as phi_consume


TextCap = Callable[[Any], Awaitable[str]] | Callable[[Any], str]
GuardedTextCap = Callable[[Any], Awaitable[Dict[str,Any]]]


def text_capability_for_user(func: Callable[[Dict[str,Any]], Awaitable[str]], *,
                             user_id: str,
                             capability_name: str,
                             cost: float | None = None):
    """
    עוטף פונקציה קורוטינית שמחזירה טקסט:
      - Caps קונקרנציה (גלובלי/משתמש/יכולת) + Rate-limit פר־יכולת
      - חיוב Φ-Budget (עם Evidences)
      - Evidences לפני/אחרי
      - Official Gate + Guard לפני החזרה
      - Fallback בטוח במקרה Reject/Budget
    """
    async def _wrapped(payload: Dict[str,Any]) -> Dict[str,Any]:
        cfg = load_config()
        t0 = time.time()
        # Evidence: התחלת exec
        current().add_evidence("capability_exec_start", {
            "source_url": f"cap://{capability_name}",
            "trust": 0.95,
            "ttl_s": 600,
            "payload": {"capability": capability_name}
        })
        g = u = c = None
        try:
            # נכנסים לשערי קונקרנציה
            g,u,c = await ASYNC_CAPS.enter(capability_name)
            # מחייבים תקציב Φ
            charged, remaining = phi_consume(capability_name, amount=cost, user_id=user_id)

            text = await func(payload)

            # אימות מקורות רשמיים (אם קיימים)
            run_official_checks(cfg)
            # אוכפים Guard גלובלי (חתימות/טריות/נאמנות/סוגי־ראיות)
            enforce_guard_before_respond(evidences=current().snapshot(), cfg=cfg)

            took = time.time() - t0
            current().add_evidence("capability_exec_done", {
                "source_url": f"cap://{capability_name}",
                "trust": 0.98,
                "ttl_s": 600,
                "payload": {"capability": capability_name, "charged": charged, "remaining": remaining, "took_s": took}
            })
            return {"text": text, "claims": current().snapshot()}
        except BudgetExceeded as be:
            fb = safe_text_fallback(reason="budget_exceeded", details={"capability": be.capability, "required": be.required, "available": be.available})
            return {"text": fb, "claims": current().snapshot(), "guard_rejected": True, "budget_exceeded": True}
        except GuardRejection as gr:
            fb = safe_text_fallback(reason=gr.reason, details=gr.details)
            return {"text": fb, "claims": current().snapshot(), "guard_rejected": True}
        finally:
            # שחרור סמפורים
            for s in (c,u,g):
                if s is not None:
                    try:
                        await s.__aexit__(None, None, None)
                    except Exception:
                        pass
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
        return await text_capability_for_user(cap, user_id=user_id)
        
   # רישום גלובלי קטן
registry = CapabilityRegistry()