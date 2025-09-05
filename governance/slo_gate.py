# governance/slo_gate.py
# -*- coding: utf-8 -*-
from perf.monitor import monitor_global

class SLOGateError(Exception): ...

def gate_p95(max_ms: float):
    p95 = monitor_global.p95()
    if p95 > max_ms:
        raise AssertionError(f"p95 too high: {p95:.1f}ms > {max_ms}ms")