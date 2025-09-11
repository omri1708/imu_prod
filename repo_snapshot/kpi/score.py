from __future__ import annotations
from typing import Dict, Any

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _inv_latency(p95_ms: float, cap_ms: float = 1500.0) -> float:
    """
    נרמול p95: נמוך=טוב. מחזיר 0..100
    """
    x = _clamp(p95_ms, 0.0, cap_ms)
    # 0ms -> 100, cap_ms -> 0
    return 100.0 * (1.0 - (x / cap_ms))

def compute_kpi(*, tests_passed: bool, p95_ms: float, ui_score: float,
                consistency_score: float, resolution_score: float) -> Dict[str, Any]:
    """
    מחזיר:
      - score (0..100 גבוה=טוב)
      - breakdown
    """
    t = 100.0 if tests_passed else 0.0
    lat = _inv_latency(p95_ms)
    ui = _clamp(ui_score, 0.0, 100.0)
    cons = _clamp(consistency_score, 0.0, 100.0)
    res = _clamp(resolution_score, 0.0, 100.0)

    # משקולות שמרניות: בדיקות ועקביות דומיננטיות
    w_t, w_lat, w_ui, w_cons, w_res = 0.28, 0.20, 0.12, 0.28, 0.12
    score = (w_t*t + w_lat*lat + w_ui*ui + w_cons*cons + w_res*res)
    return {
        "score": score,
        "breakdown": {"tests": t, "latency": lat, "ui": ui, "consistency": cons, "resolution": res},
        "weights": {"tests": w_t, "latency": w_lat, "ui": w_ui, "consistency": w_cons, "resolution": w_res},
    }

def resolution_quality_from_proof(*, contradictions_after_cut: int, base_score: float) -> float:
    """
    איכות רזולוציה — פשוטה אך אפקטיבית: ככל שנשארו פחות סתירות אחרי החיתוך, ובסיס העקביות גבוה — כנראה רזולוציה טובה.
    """
    penalty = min(40.0, 8.0 * float(max(0, contradictions_after_cut)))
    return _clamp(base_score - penalty, 0.0, 100.0)