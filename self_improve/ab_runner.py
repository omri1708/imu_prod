# imu_repo/self_improve/ab_runner.py
from __future__ import annotations
import os, time
from typing import Callable
from alerts.notifier import metrics_log, alert

Workload = Callable[[dict], None]
# Workload מקבל dict {"n":..., "lat_ms":..., "fail_every":..., "gate_denied_every":...}
# והוא אמור לכתוב metrics/alerts בהתאם.

def run_bucket(bucket: str, workload: Workload, params: dict)->None:
    os.environ["IMU_BUCKET"] = bucket
    workload(dict(params))

def simple_workload(params: dict)->None:
    n = int(params.get("n", 200))
    lat = float(params.get("lat_ms", 50.0))
    fail_every = int(params.get("fail_every", 0))
    gate_denied_every = int(params.get("gate_denied_every", 0))
    name = params.get("metric_name", "guarded_handler")
    for i in range(n):
        metrics_log(name, {"ok": True, "latency_ms": lat})
        if gate_denied_every and (i % gate_denied_every == 0):
            alert("evidence_gate_denied", severity="high", meta={})
        elif fail_every and (i % fail_every == 0):
            alert("handler_failure", severity="high", meta={})
        # מרווח קטן כדי לאחד timestamps שונים
        if (i % 50) == 0:
            time.sleep(0.001)