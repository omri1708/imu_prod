# imu_repo/runtime/metrics.py
from __future__ import annotations
from typing import Dict, Any, List
import time, threading

class _Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self._lat: Dict[str, List[float]] = {}
        self._ctr: Dict[str, int] = {}

    def reset(self) -> None:
        with self._lock:
            self._lat.clear()
            self._ctr.clear()

    def inc(self, key: str, delta: int=1) -> None:
        with self._lock:
            self._ctr[key] = self._ctr.get(key, 0) + int(delta)

    def record_latency_ms(self, key: str, ms: float, *, keep:int=1000) -> None:
        with self._lock:
            arr = self._lat.get(key)
            if arr is None:
                arr = []
                self._lat[key] = arr
            arr.append(float(ms))
            if len(arr) > keep:
                # שמירה על זיכרון
                drop = len(arr) - keep
                del arr[0:drop]

    def p95(self, key: str) -> float | None:
        with self._lock:
            arr = list(self._lat.get(key, []))
        if not arr:
            return None
        arr.sort()
        idx = int(0.95*(len(arr)-1))
        return arr[idx]

    def snapshot(self) -> Dict[str,Any]:
        with self._lock:
            return {"latencies": {k:list(v) for k,v in self._lat.items()},
                    "counters": dict(self._ctr)}

metrics = _Metrics()

class AsyncTimer:
    def __init__(self, key: str):
        self.key = key
        self._t0 = None
    async def __aenter__(self):
        self._t0 = time.perf_counter()
        return self
    async def __aexit__(self, exc_type, exc, tb):
        dt = (time.perf_counter() - self._t0)*1000.0
        metrics.record_latency_ms(self.key, dt)

def atimer(key: str) -> AsyncTimer:
    return AsyncTimer(key)