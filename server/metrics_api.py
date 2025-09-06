# server/metrics_api.py  (UPDATED)
from __future__ import annotations
from fastapi import APIRouter
from typing import Dict, Any
from runtime.p95 import GATES  # windows store sorted values
from .stream_wfq_stats import broker_stats

router = APIRouter(prefix="/metrics", tags=["metrics"])

def _percentile(vals, q):
    if not vals: return 0.0
    n=len(vals); idx=int(q*(n-1))
    return float(vals[idx])

@router.get("/summary")
def summary() -> Dict[str, Any]:
    p_snapshot={}
    for key, win in getattr(GATES, "windows", {}).items():
        vals = list(win.values)  # already sorted
        p_snapshot[key] = {
            "count": len(vals),
            "p50_ms": _percentile(vals, 0.50),
            "p90_ms": _percentile(vals, 0.90),
            "p95_ms": _percentile(vals, 0.95),
            "p99_ms": _percentile(vals, 0.99),
        }
    return {"ok": True, "latency": p_snapshot, "wfq": broker_stats()}