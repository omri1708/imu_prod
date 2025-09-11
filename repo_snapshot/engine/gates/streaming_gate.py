# imu_repo/engine/gates/streaming_gate.py
from __future__ import annotations
from typing import Dict, Any
from realtime.metrics_stream import StreamMetrics

class StreamingGate:
    """
    Gate לריל־טיים: בודק מגבלות p95 RTT וקיבולת תור (Backpressure).
      cfg = {"p95_rtt_ms_max": 120.0, "max_queue_depth": 80}
    """
    def __init__(self, metrics: StreamMetrics, *, p95_rtt_ms_max: float=200.0, max_queue_depth: int=200):
        self.metrics = metrics
        self.p95_max = float(p95_rtt_ms_max)
        self.qmax = int(max_queue_depth)

    def check(self) -> Dict[str,Any]:
        snap = self.metrics.snapshot()
        ok = (snap["p95_rtt_ms"] <= self.p95_max) and (snap["queue_depth"] <= self.qmax)
        viol=[]
        if snap["p95_rtt_ms"] > self.p95_max:
            viol.append(("p95_rtt_exceeded", snap["p95_rtt_ms"], self.p95_max))
        if snap["queue_depth"] > self.qmax:
            viol.append(("queue_depth_exceeded", snap["queue_depth"], self.qmax))
        return {"ok": ok, "snapshot": snap, "violations": viol}