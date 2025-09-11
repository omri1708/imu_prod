# synth/canary.py
from __future__ import annotations
from typing import Dict, Any

def compare_kpis(baseline: Dict[str,Any], candidate: Dict[str,Any], max_latency_ms: float = 200.0) -> Dict[str,Any]:
    """
    Compare trivial KPIs: all tests passed; elapsed within threshold vs baseline.
    """
    ok = candidate.get("passed", False)
    verdict = ok and (candidate.get("avg_latency_ms", 0.0) <= max(baseline.get("avg_latency_ms", 9999.0), max_latency_ms))
    return {"ok": bool(verdict), "reason": None if verdict else "kpi_threshold"}