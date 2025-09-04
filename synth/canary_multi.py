# imu_repo/synth/canary_multi.py
from __future__ import annotations
from typing import Dict, Any, List

def _pass_stage(stage: Dict[str, Any], *, baseline_kpi: float, candidate_kpi: float) -> bool:
    """
    שלב עובר אם:
      - candidate_kpi ≥ min_score
      - candidate_kpi ≥ baseline_kpi - max_regression
    """
    min_score = float(stage.get("min_score", 70.0))
    max_reg = float(stage.get("max_regression", 5.0))  # באחוזים
    ok_score = candidate_kpi >= min_score
    ok_reg = (candidate_kpi + 1e-6) >= (baseline_kpi * (1.0 - max_reg/100.0))
    return bool(ok_score and ok_reg)

def run_staged_canary(*, baseline_kpi: float, candidate_kpi: float,
                      stages: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """
    stages: רשימת שלבים עם:
      - name
      - traffic_percent
      - min_score
      - max_regression (%)
    """
    stages = stages or [
        {"name":"shadow", "traffic_percent":0,  "min_score":65, "max_regression":10},
        {"name":"1pct",   "traffic_percent":1,  "min_score":68, "max_regression": 8},
        {"name":"5pct",   "traffic_percent":5,  "min_score":70, "max_regression": 6},
        {"name":"25pct",  "traffic_percent":25, "min_score":72, "max_regression": 5},
        {"name":"100pct", "traffic_percent":100,"min_score":75, "max_regression": 4},
    ]
    results: List[Dict[str,Any]] = []
    approved = True
    for st in stages:
        ok = _pass_stage(st, baseline_kpi=baseline_kpi, candidate_kpi=candidate_kpi)
        results.append({"stage": st["name"], "ok": ok, "min_score": st["min_score"],
                        "max_regression": st["max_regression"], "traffic_percent": st["traffic_percent"]})
        if not ok:
            approved = False
            break
    return {"approved": approved, "stages": results, "baseline_kpi": baseline_kpi, "candidate_kpi": candidate_kpi}