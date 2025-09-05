# governance/slo_gate.py
# -*- coding: utf-8 -*-
from perf.monitor import monitor_global

class SLOGateError(Exception): ...

def gate_p95(max_ms: float):
    p95 = monitor_global.p(0.95)
    if p95 > max_ms:
        raise SLOGateError(f"p95_exceeded:{p95:.2f}>{max_ms:.2f}")