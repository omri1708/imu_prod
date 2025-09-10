from __future__ import annotations
from typing import Dict, Any, List

DEFAULT_STAGES = [
    {"stage":"shadow", "duration_s":60},
    {"stage":"1pct",  "duration_s":120},
    {"stage":"10pct", "duration_s":180}
]

def run_canary(get_kpi, stages: List[Dict[str,Any]] = None) -> Dict[str,Any]:
    """
    get_kpi(stage_dict) -> {"p95_ms":..., "error_rate":..., "cost_usd":..., "target_ms":..., "target_err":..., "target_cost":...}
    """
    stages = list(stages or DEFAULT_STAGES)
    out = {"stages": []}
    for s in stages:
        k = get_kpi(s)
        ok = (
            float(k.get("p95_ms",1e9))   <= float(k.get("target_ms",1500.0)) and
            float(k.get("error_rate",1)) <= float(k.get("target_err",0.02))   and
            float(k.get("cost_usd",1e9)) <= float(k.get("target_cost",  0.02))
        )
        out["stages"].append({"stage": s["stage"], "kpi": k, "ok": ok})
        if not ok:
            return {"ok": False, **out}
    return {"ok": True, **out}
