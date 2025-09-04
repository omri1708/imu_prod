# imu_repo/orchestrator/orchestrator.py
from __future__ import annotations
from typing import Dict, Any, List
import os, json, time, uuid, random, threading

from orchestrator.consensus import LeaderElector
from orchestrator.registry import list_workers, healthy
from caps.queue import FileQueue
from runtime.metrics import metrics

TASKS_Q_DIR = "/mnt/data/imu_repo/queues/tasks"
WORKERS_ROOT = "/mnt/data/imu_repo/queues/workers"
RESULTS_Q_DIR = "/mnt/data/imu_repo/queues/results"

os.makedirs(TASKS_Q_DIR, exist_ok=True)
os.makedirs(WORKERS_ROOT, exist_ok=True)
os.makedirs(RESULTS_Q_DIR, exist_ok=True)

class Orchestrator:
    def __init__(self, *, node_id: str | None=None, hb_s: float=1.0):
        self.node_id = node_id or uuid.uuid4().hex[:12]
        self.elector = LeaderElector(self.node_id, ttl_s=8.0)
        self.tasks = FileQueue(TASKS_Q_DIR)
        self.results = FileQueue(RESULTS_Q_DIR)
        self._stop = threading.Event()
        self._hb_s = float(hb_s)
        self._rr: Dict[str,int] = {}  # round-robin pointers per capability

    def stop(self): self._stop.set()

    def _choose_worker(self, capability: str) -> str | None:
        ws = healthy(list_workers(), max_age_s=6.0)
        cand = [w for w in ws if capability in (w.get("capabilities") or [])]
        if not cand: return None
        idx = self._rr.get(capability, 0) % len(cand)
        self._rr[capability] = idx + 1
        return cand[idx]["worker_id"]

    def _deliver(self, worker_id: str, payload: Dict[str,Any]) -> None:
        from caps.queue import FileQueue
        q = FileQueue(os.path.join(WORKERS_ROOT, worker_id))
        q.enqueue(payload)

    def _dispatch_loop(self):
        while not self._stop.is_set():
            if not self.elector.is_leader():
                time.sleep(0.05)
                continue
            msg = self.tasks.dequeue()
            if not msg:
                time.sleep(0.02)
                self.elector.renew()
                continue
            try:
                task = msg.get("task"); task_id = msg.get("task_id")
                wid = self._choose_worker(task)
                if wid is None:
                    # אין worker מתאים — ננסה מאוחר יותר
                    time.sleep(0.05)
                    continue
                t0 = time.perf_counter()
                self._deliver(wid, msg)
                dt = (time.perf_counter() - t0)*1000.0
                metrics.record_latency_ms("orchestrator.dispatch", dt)
            finally:
                # ack רק אחרי deliver — אם אין worker נשאיר בתור
                self.tasks.ack(msg)

    def _leadership_loop(self):
        while not self._stop.is_set():
            if self.elector.is_leader():
                self.elector.renew()
                time.sleep(self._hb_s/2)
            else:
                self.elector.try_acquire()
                time.sleep(self._hb_s)

    def run(self):
        t1 = threading.Thread(target=self._leadership_loop, daemon=True); t1.start()
        t2 = threading.Thread(target=self._dispatch_loop, daemon=True); t2.start()
        try:
            while not self._stop.is_set():
                time.sleep(0.1)
        finally:
            self._stop.set()

def enqueue_task(task: str, args: Dict[str,Any]) -> str:
    q = FileQueue(TASKS_Q_DIR)
    task_id = uuid.uuid4().hex
    q.enqueue({"task_id": task_id, "task": task, "args": dict(args or {})})
    return task_id

def collect_results(timeout_s: float=5.0) -> List[Dict[str,Any]]:
    """
    קורא תוצאות מצטברות במשך timeout_s.
    """
    q = FileQueue(RESULTS_Q_DIR)
    out=[]
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        m = q.dequeue()
        if not m:
            time.sleep(0.02)
            continue
        out.append(m)
        q.ack(m)
    return out