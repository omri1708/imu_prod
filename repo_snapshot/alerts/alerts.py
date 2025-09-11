# alerts/alerts.py
from __future__ import annotations
import json, os
from typing import Dict, Any

def evaluate(summary: Dict[str,Any], out_dir: str) -> Dict[str,Any]:
    alerts=[]
    perf = summary.get("perf", {})
    ver  = summary.get("verify", {})
    if perf.get("p95_ms", 0) > 500.0:
        alerts.append({"kind":"perf_p95", "severity":"warn", "p95": perf["p95_ms"]})
    if not ver.get("ok", False):
        alerts.append({"kind":"contract_violation", "severity":"error", "violations": ver.get("violations", [])})
    out={"alerts":alerts}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir,"alerts.json"),"w",encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out