# synth/plan.py
from __future__ import annotations
from typing import Dict, Any, List
from synth.specs import BuildSpec, Contract

def plan(spec: BuildSpec) -> Dict[str,Any]:
    """
    Produce a simple DAG (list of steps in order) — deterministic.
    """
    steps: List[Dict[str,Any]] = [
        {"step":"generate", "desc":"Generate source code from spec"},
        {"step":"unit_tests", "desc":"Generate & run unit tests for endpoints"},
        {"step":"start_service", "desc":"Start service and probe"},
        {"step":"contract_check", "desc":"Validate outputs against contracts"},
        {"step":"package", "desc":"Tar sources into artifact"},
        {"step":"canary", "desc":"Run canary vs baseline (synthetic)"},
        {"step":"rollout", "desc":"Gate rollout by KPIs"}
    ]
    return {"kind": spec.kind, "steps": steps}