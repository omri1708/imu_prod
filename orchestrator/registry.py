# imu_repo/orchestrator/registry.py
from __future__ import annotations
from typing import Dict, Any, List
import os, json, time, uuid

ROOT = "/mnt/data/imu_repo/registry/workers"
os.makedirs(ROOT, exist_ok=True)

def worker_dir(worker_id: str) -> str:
    return os.path.join(ROOT, worker_id)

def register(capabilities: List[str], *, worker_id: str | None=None) -> str:
    wid = worker_id or uuid.uuid4().hex[:12]
    wdir = worker_dir(wid)
    os.makedirs(wdir, exist_ok=True)
    with open(os.path.join(wdir,"capabilities.json"),"w",encoding="utf-8") as f:
        json.dump({"worker_id": wid, "capabilities": list(capabilities)}, f, ensure_ascii=False)
    heartbeat(wid)
    return wid

def heartbeat(worker_id: str) -> None:
    wdir = worker_dir(worker_id)
    os.makedirs(wdir, exist_ok=True)
    with open(os.path.join(wdir,"status.json"),"w",encoding="utf-8") as f:
        json.dump({"worker_id": worker_id, "ts": time.time()}, f)

def list_workers() -> List[Dict[str,Any]]:
    out=[]
    for name in os.listdir(ROOT):
        wdir = worker_dir(name)
        try:
            caps = json.load(open(os.path.join(wdir,"capabilities.json"),"r",encoding="utf-8"))
            st   = json.load(open(os.path.join(wdir,"status.json"),"r",encoding="utf-8"))
            caps["ts"] = st.get("ts", 0.0)
            out.append(caps)
        except Exception:
            continue
    return out

def healthy(workers: List[Dict[str,Any]], *, max_age_s: float=6.0) -> List[Dict[str,Any]]:
    now = time.time()
    return [w for w in workers if (now - float(w.get("ts",0.0))) <= max_age_s]