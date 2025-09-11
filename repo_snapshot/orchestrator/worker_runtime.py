# imu_repo/orchestrator/worker_runtime.py
from __future__ import annotations
from typing import Dict, Any, List
import os, json, time, threading

from orchestrator.registry import register, heartbeat
from caps.queue import FileQueue
from runtime.metrics import metrics

WORKERS_ROOT = "/mnt/data/imu_repo/queues/workers"
RESULTS_Q_DIR = "/mnt/data/imu_repo/queues/results"

os.makedirs(WORKERS_ROOT, exist_ok=True)
os.makedirs(RESULTS_Q_DIR, exist_ok=True)

class Worker:
    def __init__(self, capabilities: List[str], *, worker_id: str | None=None, hb_interval_s: float=1.0):
        self.capabilities = list(capabilities)
        self.worker_id = register(self.capabilities, worker_id=worker_id)
        self.queue = FileQueue(os.path.join(WORKERS_ROOT, self.worker_id))
        self.results = FileQueue(RESULTS_Q_DIR)
        self.hb_interval_s = float(hb_interval_s)
        self._stop = threading.Event()

    def stop(self): self._stop.set()

    def _hb_loop(self):
        while not self._stop.is_set():
            heartbeat(self.worker_id)
            time.sleep(self.hb_interval_s)

    def _handle(self, msg: Dict[str,Any]) -> Dict[str,Any]:
        task = msg.get("task")
        args = msg.get("args", {})
        from caps.tasks.basic import run_task
        try:
            res = run_task(task, args)
            return {"task_id": msg.get("task_id"), "ok": True, "task": task, "result": res}
        except Exception as e:
            return {"task_id": msg.get("task_id"), "ok": False, "task": task, "error": str(e)}

    def run(self):
        t = threading.Thread(target=self._hb_loop, daemon=True); t.start()
        try:
            while not self._stop.is_set():
                msg = self.queue.dequeue()
                if not msg:
                    time.sleep(0.05)
                    continue
                out = self._handle(msg)
                self.results.enqueue(out)
                # ack למסר
                self.queue.ack(msg)
                # מדד בוצע
                from runtime.metrics import metrics
                metrics.inc(f"worker.{self.worker_id}.completed", 1)
        finally:
            self._stop.set()