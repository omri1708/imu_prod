# server/runbook_history.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from pathlib import Path
import json, time, uuid, threading

HIST_DIR = Path(".imu/runbook/history")
HIST_DIR.mkdir(parents=True, exist_ok=True)
_LOCK = threading.Lock()

@dataclass
class HistoryRecord:
    run_id: str
    flow: str
    params: Dict[str, Any]
    ts_start: float
    ts_end: Optional[float] = None
    events: List[Dict[str,Any]] = None
    result: Optional[Dict[str,Any]] = None

def _path(run_id: str) -> Path:
    return HIST_DIR / f"{run_id}.json"

def record_start(flow: str, params: Dict[str,Any]) -> str:
    run_id = f"{flow}-{uuid.uuid4().hex[:8]}-{int(time.time())}"
    rec = HistoryRecord(run_id=run_id, flow=flow, params=params, ts_start=time.time(), events=[])
    with _LOCK:
        _path(run_id).write_text(json.dumps(asdict(rec), ensure_ascii=False, indent=2), encoding="utf-8")
    return run_id

def append_event(run_id: str, event: Dict[str,Any]):
    p = _path(run_id)
    with _LOCK:
        obj = json.loads(p.read_text(encoding="utf-8"))
        obj["events"].append(dict(event))
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def finalize(run_id: str, result: Dict[str,Any]):
    p = _path(run_id)
    with _LOCK:
        obj = json.loads(p.read_text(encoding="utf-8"))
        obj["ts_end"] = time.time()
        obj["result"] = result
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def list_history() -> List[Dict[str,Any]]:
    items=[]
    for f in HIST_DIR.glob("*.json"):
        try:
            o=json.loads(f.read_text(encoding="utf-8"))
            items.append({"run_id":o["run_id"],"flow":o["flow"],"ts_start":o["ts_start"],"ts_end":o.get("ts_end"),"path":str(f)})
        except Exception: pass
    return sorted(items, key=lambda x: x["ts_start"], reverse=True)

def load_history(run_id_or_path: str) -> Dict[str,Any]:
    p = Path(run_id_or_path)
    if not p.exists():
        p = _path(run_id_or_path)
    return json.loads(p.read_text(encoding="utf-8"))