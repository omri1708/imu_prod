# imu_repo/realtime/metrics_stream.py
from __future__ import annotations
from typing import List, Dict, Any, Deque, Tuple
from collections import deque
import time
import statistics

class StreamMetrics:
    """
    חלון הזזה למדידת RTT (מ״ש), קצב הודעות/בתים לשנייה, ועומס התור (backlog).
    """
    def __init__(self, window_s: float=30.0):
        self.window_s = float(window_s)
        self.rtts: Deque[Tuple[float,float]] = deque()  # (ts, rtt_ms)
        self.bytes_in: Deque[Tuple[float,int]] = deque()
        self.bytes_out: Deque[Tuple[float,int]] = deque()
        self.queue_depth = 0

    def _trim(self, dq: Deque[Tuple[float, float|int]]):
        t0 = time.time() - self.window_s
        while dq and dq[0][0] < t0:
            dq.popleft()

    def record_rtt_ms(self, rtt_ms: float):
        self.rtts.append((time.time(), float(rtt_ms))); self._trim(self.rtts)

    def record_in(self, nbytes: int):
        self.bytes_in.append((time.time(), int(nbytes))); self._trim(self.bytes_in)

    def record_out(self, nbytes: int):
        self.bytes_out.append((time.time(), int(nbytes))); self._trim(self.bytes_out)

    def set_queue_depth(self, depth: int):
        self.queue_depth = int(depth)

    def p95_rtt_ms(self) -> float:
        vals = [v for _,v in self.rtts]
        if not vals: return 0.0
        vals.sort()
        k = int(0.95*(len(vals)-1))
        return float(vals[k])

    def rate_in_bps(self) -> float:
        return self._rate_bps(self.bytes_in)

    def rate_out_bps(self) -> float:
        return self._rate_bps(self.bytes_out)

    def _rate_bps(self, dq: Deque[Tuple[float,int]]) -> float:
        if not dq: return 0.0
        total = sum(v for _,v in dq)
        dur = max(1e-3, dq[-1][0]-dq[0][0])
        return total/dur

    def snapshot(self) -> Dict[str,Any]:
        return {
            "p95_rtt_ms": self.p95_rtt_ms(),
            "rate_in_bps": self.rate_in_bps(),
            "rate_out_bps": self.rate_out_bps(),
            "queue_depth": self.queue_depth
        }