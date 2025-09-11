# imu_repo/kpi/policy_adapter.py
from __future__ import annotations
from typing import Dict, Any
from kpi.score import _clamp, _inv_latency

def compute_kpi_with_policy(*, tests_passed: bool, p95_ms: float, ui_score: float,
                            consistency_score: float, resolution_score: float,
                            weights: Dict[str, float]) -> Dict[str, Any]:
    t = 100.0 if tests_passed else 0.0
    lat = _inv_latency(p95_ms)
    ui = _clamp(ui_score, 0.0, 100.0)
    cons = _clamp(consistency_score, 0.0, 100.0)
    res  = _clamp(resolution_score, 0.0, 100.0)

    w_t   = float(weights.get("tests", 0.28))
    w_lat = float(weights.get("latency", 0.20))
    w_ui  = float(weights.get("ui", 0.12))
    w_cons= float(weights.get("consistency", 0.28))
    w_res = float(weights.get("resolution", 0.12))
    norm  = max(1e-9, (w_t+w_lat+w_ui+w_cons+w_res))
    w_t, w_lat, w_ui, w_cons, w_res = (w_t/norm, w_lat/norm, w_ui/norm, w_cons/norm, w_res/norm)

    score = (w_t*t + w_lat*lat + w_ui*ui + w_cons*cons + w_res*res)
    return {
        "score": score,
        "breakdown": {"tests": t, "latency": lat, "ui": ui, "consistency": cons, "resolution": res},
        "weights": {"tests": w_t, "latency": w_lat, "ui": w_ui, "consistency": w_cons, "resolution": w_res},
    }