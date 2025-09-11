# imu_repo/dist/worker.py
from __future__ import annotations
from typing import Callable, Dict, Any, Tuple
import os, time, multiprocessing as mp
from dist.job_queue import reserve, ack, nack

def _safe_call(fn: Callable[[Dict[str,Any]], Tuple[bool, Dict[str,Any] | None, Dict[str,Any] | None]], job: Dict[str,Any]):
    """
    fn(payload) -> (ok, result, compensate)
    compensate משמש ל-rollback אם כשל.
    """
    try:
        return fn(job["payload"])
    except Exception as e:
        return (False, {"error": str(e)}, {"type":"noop"})

def worker_loop(name: str, fn: Callable[[Dict[str,Any]], Tuple[bool, Dict[str,Any] | None, Dict[str,Any] | None]], *, stop_after_idle_s: float=2.0):
    idle_t0 = time.time()
    while True:
        job = reserve()
        if not job:
            if time.time()-idle_t0 > stop_after_idle_s:
                break
            time.sleep(0.1); continue
        idle_t0 = time.time()
        ok, res, comp = _safe_call(fn, job)
        if ok:
            ack(job["job_id"], result=res or {"ok":True})
        else:
            nack(job["job_id"], reason=res.get("error","failed"), compensate=comp)

def start_pool(n: int, fn: Callable[[Dict[str,Any]], Tuple[bool, Dict[str,Any] | None, Dict[str,Any] | None]]) -> list[mp.Process]:
    procs=[]
    for i in range(n):
        p = mp.Process(target=worker_loop, args=(f"w{i+1}", fn), kwargs={"stop_after_idle_s": 1.5}, daemon=True)
        p.start(); procs.append(p)
    return procs

def join_pool(procs: list[mp.Process]) -> None:
    for p in procs: p.join()