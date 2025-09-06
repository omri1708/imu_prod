# server/metrics_api.py
# FastAPI Router למדדים: p95 per-key, סטטיסטיקות WFQ, ו-state של נושאים פעילים.
from __future__ import annotations
from fastapi import APIRouter
from typing import Dict, Any
from runtime.p95 import GATES
from .stream_wfq_stats import broker_stats

router = APIRouter(prefix="/metrics", tags=["metrics"])

@router.get("/summary")
def summary() -> Dict[str, Any]:
    # p95 (צילום מצב) — שומר את הערכים המחושבים ללא התאפסות
    p95_snapshot = {}
    for key, win in getattr(GATES, "windows", {}).items():
        p95_snapshot[key] = {"count": len(win.values), "p95_ms": win.p95()}
    return {
        "ok": True,
        "p95": p95_snapshot,
        "wfq": broker_stats(),
    }