# server/decision_log.py
# רישום החלטות Gatekeeper לקובץ JSON Lines שנכנס ל-Bundle המאוחד.
from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
import json, time

DEC_DIR = Path(".imu/decisions")
DEC_DIR.mkdir(parents=True, exist_ok=True)

def record_gate_decision(run_id: str, stage: str, gate_input: Dict[str,Any], result: Dict[str,Any]) -> str:
    rec={
        "ts": time.time(),
        "run_id": run_id,
        "stage": stage,           # e.g. "pre-promote"
        "gate_input": gate_input, # evidences/checks/p95
        "result": result          # {"ok":bool, "reasons":[...]}
    }
    path = DEC_DIR / f"{run_id}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False)+"\n")
    return str(path)