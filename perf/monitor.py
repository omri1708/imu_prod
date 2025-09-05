# perf/monitor.py
# -*- coding: utf-8 -*-
import time, threading
from typing import List
import threading, bisect

class PerfMonitor:
    def __init__(self, window_size=2048):
        self.lock = threading.Lock()
        self.samples: List[float] = []
        self.window = window_size

    def observe_ms(self, ms: float):
        with self.lock:
            self.samples.append(ms)
            if len(self.samples) > self.window:
                self.samples = self.samples[-self.window:]

    def p95_ms(self) -> float:
        with self.lock:
            if not self.samples: return 0.0
            arr = sorted(self.samples)
            idx = int(0.95 * (len(arr)-1))
            return float(arr[idx])


class _Hdr:
    def __init__(self):
        self._lock = threading.Lock()
        self._vals = []

    def observe(self, v_ms: float):
        with self._lock:
            bisect.insort(self._vals, v_ms)
            # שמירה רזה
            if len(self._vals) > 5000:
                self._vals = self._vals[-2500:]

    def p(self, q: float) -> float:
        with self._lock:
            if not self._vals:
                return 0.0
            idx = int(q * (len(self._vals)-1))
            return self._vals[idx]


monitor_global = _Hdr()

monitor_global = PerfMonitor()