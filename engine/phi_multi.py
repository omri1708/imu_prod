# imu_repo/engine/phi_multi.py
from __future__ import annotations
from typing import Dict, Any

DEFAULT_WEIGHTS = {
    "latency": 0.6,     # p95_ms נמוך עדיף
    "cost":    0.25,    # cost_units (למשל אורך קוד/כח חישוב)
    "errors":  0.10,    # error_rate
    "distrust":0.03,    # 1 - source_trust
    "energy":  0.015,   # energy_units
    "memory":  0.005,   # mem_kb
}

# סולמות נרמול (הופכים מטריקות ל-[0,1] בקירוב; גמיש אך דטרמיניסטי)
NORM = {
    "latency_ms": 1000.0,      # 1000ms → 1.0
    "cost_units": 10000.0,     # 10k תווים/יחידות → 1.0
    "error_rate": 1.0,         # כבר [0,1]
    "distrust":   1.0,         # 1 - trust
    "energy":     100.0,       # יחידות אנרגיה יחסיות
    "mem_kb":     1024.0,      # 1MB → 1.0
}

def clamp01(x: float) -> float:
    return 0.0 if x <= 0 else (1.0 if x >= 1.0 else x)

def normalize_metrics(perf: Dict[str, float]) -> Dict[str,float]:
    lat = clamp01(float(perf.get("p95_ms", 0.0)) / NORM["latency_ms"])
    cost = clamp01(float(perf.get("cost_units", 0.0)) / NORM["cost_units"])
    err = clamp01(float(perf.get("error_rate", 0.0)) / NORM["error_rate"])
    distrust = clamp01(1.0 - float(perf.get("source_trust", 1.0)))
    energy = clamp01(float(perf.get("energy_units", 0.0)) / NORM["energy"])
    mem = clamp01(float(perf.get("mem_kb", 0.0)) / NORM["mem_kb"])
    return {
        "latency": lat,
        "cost": cost,
        "errors": err,
        "distrust": distrust,
        "energy": energy,
        "memory": mem,
    }

def phi_score(perf: Dict[str, float], weights: Dict[str,float] | None = None) -> float:
    """
    Φ מינימיזציה: קטן יותר טוב. 
    perf חייב להכיל: p95_ms, cost_units, error_rate, source_trust, energy_units, mem_kb.
    """
    ws = dict(DEFAULT_WEIGHTS)
    if weights:
        ws.update({k: float(v) for k,v in weights.items() if k in ws})
    nm = normalize_metrics(perf)
    # סכימה משוקללת
    phi = 0.0
    for k, w in ws.items():
        phi += float(w) * float(nm.get(k, 0.0))
    return float(phi)