# imu_repo/engine/phi.py
from __future__ import annotations
from typing import Dict, Any, Tuple
import math

_DEFAULT = {
    "w_p95": 1.0,       # לכל ms
    "w_error": 10_000,  # ענישת שגיאה גבוהה
    "w_cost": 0.1,      # יחידות עלות לוגיות
    "max_allowed": 5_000.0  # סף גג לרולבאק (התאמה לפרוד)
}

def compute_phi(metrics: Dict[str,Any], weights: Dict[str,float]|None=None) -> float:
    w = dict(_DEFAULT)
    if weights:
        w.update(weights)
    p95 = float(metrics.get("p95_ms", 0.0))
    err = float(metrics.get("error_rate", 0.0))  # 0..1
    cost = float(metrics.get("cost_units", 0.0))
    phi = w["w_p95"]*p95 + w["w_error"]*err + w["w_cost"]*cost
    # הגנה: NaN/∞
    if math.isnan(phi) or math.isinf(phi):
        return float("inf")
    return phi

def is_better(phi_new: float, phi_old: float, *, eps: float=1e-9) -> bool:
    return (phi_new + eps) < phi_old

def max_allowed(weights: Dict[str,float]|None=None) -> float:
    w = dict(_DEFAULT)
    if weights:
        w.update(weights)
    return float(w["max_allowed"])