# engine/pipelines/compat.py
from __future__ import annotations
import asyncio
from typing import Any, Dict
from engine.pipelines.orchestrator import Orchestrator, default_runners

_orch = Orchestrator(default_runners())

def run_pipeline_compat(spec_or_prog: Any,
                        user_id: str = "anon",
                        ctx: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    חתימה "ישנה" מסונכרנת—בפנים מריץ Orchestrator אסינכרוני.
    אפשר להחליף בהדרגה קריאות ישנות לקריאה לפונקציה הזו.
    """
    _ctx = dict(ctx or {})
    _ctx.setdefault("user_id", user_id)
    return asyncio.run(_orch.run_any(spec_or_prog, _ctx))
