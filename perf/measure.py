# perf/measure.py
from __future__ import annotations
import time, threading, http.client
from typing import Dict, Any, List
import time, statistics


class PerfWindow:
    def __init__(self, capacity:int=200):
        self.capacity = capacity
        self.samples: List[float] = []

    def add(self, ms:float):
        self.samples.append(ms)
        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def p95(self) -> float:
        if not self.samples:
            return 0.0
        s = sorted(self.samples)
        idx = int(0.95*(len(s)-1))
        return s[idx]

class PerfRegistry:
    def __init__(self):
        self._by_key: Dict[str,PerfWindow] = {}

    def track(self, key:str, elapsed_ms:float):
        self._by_key.setdefault(key, PerfWindow()).add(elapsed_ms)

    def summary(self) -> Dict[str,Any]:
        return {k: {"count":len(w.samples), "p95_ms":w.p95()} for k,w in self._by_key.items()}

BUILD_PERF = PerfWindow()
JOB_PERF   = PerfWindow()

def measure(fn, *args, **kwargs):
    t0=time.time(); out=fn(*args, **kwargs); dt=time.time()-t0
    return out, dt


def _one(port: int, path: str="/") -> float:
    t0=time.time()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3.0)
    conn.request("GET", path)
    r = conn.getresponse(); r.read()
    return (time.time()-t0)*1000.0


def load_test(port: int, paths: List[str], concurrency: int = 8, total_requests: int = 40) -> Dict[str,Any]:
    lat: List[float] = []
    lock = threading.Lock()
    idx = {"i": 0}
    def worker():
        while True:
            with lock:
                if idx["i"] >= total_requests: return
                i = idx["i"]; idx["i"] += 1
                path = paths[i % len(paths)]
            try:
                ms = _one(port, path)
                with lock: lat.append(ms)
            except Exception:
                with lock: lat.append(3000.0)  # timeout sentinel
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    for t in threads: t.start()
    for t in threads: t.join()
    lat.sort()
    def pct(p):
        if not lat: return 0.0
        k = max(0, min(len(lat)-1, int(round((p/100.0)*(len(lat)-1)))))
        return lat[k]
    return {
        "n": len(lat),
        "avg_ms": statistics.mean(lat) if lat else 0.0,
        "p50_ms": pct(50),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
        "samples": lat[:50]
    }
