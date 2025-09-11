# imu_repo/obs/alerts.py
from __future__ import annotations
from typing import Dict, Any, List, Callable, Optional
import time, json, os, threading
from obs.kpi import KPI
from persistence.policy_store import PolicyStore
from obs.tracing import Tracer


class Alert:
    def __init__(self, name: str, predicate: Callable[[Dict[str,Any]], bool], notify: Callable[[str,Dict[str,Any]],None]):
        self.name = name
        self.predicate = predicate
        self.notify = notify

class AlertEngine:
    """Simple rule-based alert engine."""

    def __init__(self):
        self.rules: List[Alert] = []

    def add_rule(self, rule: Alert):
        self.rules.append(rule)

    def evaluate(self, snapshot: Dict[str,Any]):
        for r in self.rules:
            try:
                if r.predicate(snapshot):
                    r.notify(r.name, snapshot)
            except Exception:
                # alerts engine is best-effort; shouldn't crash the process
                pass


class AlertSink:
    def __init__(self, path: str = ".imu_state/alerts.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path,"w",encoding="utf-8"): pass
    def push(self, alert: Dict[str,Any]):
        with open(self.path,"a",encoding="utf-8") as f:
            f.write(json.dumps(alert, ensure_ascii=False)+"\n")


class AlertMonitor:
    """
    מנטר KPI ומייצר התראות בזמן אמת ע"פ ספי מדיניות:
      thresholds: {max_error_rate, max_p95_latency_ms}
    """
    def __init__(self, kpi: KPI, policy: PolicyStore, sink: Optional[AlertSink]=None, period_s: float = 2.0):
        self.kpi = kpi
        self.policy = policy
        self.sink = sink or AlertSink()
        self.period = period_s
        self.tracer = Tracer()
        self._stop = False
        self._thr : Optional[threading.Thread] = None

    def start(self):
        if self._thr and self._thr.is_alive(): return
        self._stop = False
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def stop(self):
        self._stop = True
        if self._thr: self._thr.join(timeout=2.0)

    def _loop(self):
        while not self._stop:
            snap = self.kpi.snapshot()
            pol  = self.policy.current().get("config",{}).get("thresholds",{
                "max_error_rate": 0.02, "max_p95_latency_ms": 800.0
            })
            alerts=[]
            if snap["error_rate"] > pol["max_error_rate"]:
                alerts.append({"kind":"error_rate", "value":snap["error_rate"], "threshold":pol["max_error_rate"]})
            if snap["p95"] > pol["max_p95_latency_ms"]:
                alerts.append({"kind":"p95_latency", "value":snap["p95"], "threshold":pol["max_p95_latency_ms"]})
            for a in alerts:
                rec={"ts":time.time(),"alert":a,"kpi":snap}
                self.sink.push(rec)
                self.tracer.emit("alert", rec)
            time.sleep(self.period)