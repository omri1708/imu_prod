# engine/adapters/facade.py
from __future__ import annotations
from typing import Any, Dict

from engine.pipeline_bindings import run_adapter as run_v1   # (1)
from engine.adapter_registry import get_adapter as get_v2    # משמש ב-(3),(7)
class AdapterFacade:
    def build(self, job: Dict[str,Any], user: str, ws: str, policy, ev_index):
        kind = job.get("kind")
        try:
            ad = get_v2(kind)
            return ad.build(job, user, ws, policy, ev_index)
        except Exception:
            # v1 fallback: מצפה kwargs שונים; מתאים שדות בסיסיים
            return run_v1(kind, **job)
