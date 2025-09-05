# imu_repo/engine/policy_drilldown.py
from __future__ import annotations
import os, json, time
from typing import Dict, Any, List, Optional, Tuple, Iterable
from collections import defaultdict

def _audit_dir() -> str:
    d = os.environ.get("IMU_AUDIT_DIR") or ".audit"
    os.makedirs(d, exist_ok=True); return d

def _read_jsonl(path: str) -> Iterable[Dict[str,Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def load_rollout_history(fname: str="rollout_orchestrator.jsonl") -> List[Dict[str,Any]]:
    return list(_read_jsonl(os.path.join(_audit_dir(), fname)))

def drilldown_by_stage(history: List[Dict[str,Any]]) -> Dict[str,Any]:
    by_stage: Dict[str,Any] = defaultdict(lambda: {"passes":0,"fails":0,"near_miss":0,"headrooms":[]})
    for ev in history:
        st = ev.get("stage") or ev.get("final_stage") or "unknown"
        rec = by_stage[st]
        if ev.get("evt") == "autotune":
            # נשמר בהיסטוריית stage אחרת — נתעלם כאן
            continue
        gate = ev.get("gate")
        if gate == "pass":
            rec["passes"] += 1
            hr = ev.get("perf_headroom")
            if isinstance(hr, (int,float)):
                rec["headrooms"].append(float(hr))
            nm = ev.get("near_miss")
            if nm:
                rec["near_miss"] += 1
        elif gate == "fail":
            rec["fails"] += 1
    return by_stage

def summarize(history: List[Dict[str,Any]]) -> Dict[str,Any]:
    stages = drilldown_by_stage(history)
    worst_stage = None
    worst_avg_hr = float("inf")
    for name, rec in stages.items():
        hrs = rec["headrooms"]
        avg = sum(hrs)/len(hrs) if hrs else float("inf")
        if avg < worst_avg_hr:
            worst_avg_hr = avg; worst_stage = name
    return {
        "stages": stages,
        "worst_stage": worst_stage,
        "worst_avg_headroom": (None if worst_avg_hr == float("inf") else worst_avg_hr)
    }