# imu_repo/service_mesh/health.py
from __future__ import annotations
from typing import Dict, Any
import time, asyncio
from runtime.metrics import metrics
from runtime.async_sandbox import SandboxRuntime, PolicyError

class BackendState:
    def __init__(self, name: str, host: str, port: int,
                 *, ewma_alpha: float=0.2, max_ewma_ms: float=1500.0,
                 max_inflight: int=64, fail_open_s: float=6.0):
        self.name = name
        self.host = host
        self.port = int(port)
        self.ewma_alpha = float(ewma_alpha)
        self.max_ewma_ms = float(max_ewma_ms)
        self.max_inflight = int(max_inflight)
        self.fail_open_s = float(fail_open_s)

        self.ewma_ms: float | None = None
        self.inflight = 0
        self.healthy = True
        self.last_ok = 0.0
        self.breaker_open_until = 0.0
        self.failures = 0
        self.success = 0

    def record_latency(self, ms: float) -> None:
        if self.ewma_ms is None:
            self.ewma_ms = float(ms)
        else:
            self.ewma_ms = (1.0 - self.ewma_alpha)*self.ewma_ms + self.ewma_alpha*float(ms)

    def mark_ok(self) -> None:
        self.healthy = True
        self.last_ok = time.time()
        self.failures = 0
        self.success += 1

    def mark_fail(self) -> None:
        self.failures += 1
        if self.failures >= 3:
            # פתח circuit ל-few seconds
            self.breaker_open_until = time.time() + self.fail_open_s
        self.healthy = False

    def circuit_open(self) -> bool:
        return time.time() < self.breaker_open_until

    def load_shed(self) -> bool:
        if self.inflight >= self.max_inflight:
            return True
        if self.ewma_ms is not None and self.ewma_ms > self.max_ewma_ms:
            return True
        return False

    def score(self) -> float:
        """
        ניקוד לבחירה: בריא? כמה עומס? כמה EWMA קטן? כמה זמן עבר מאז תקין?
        גבוה=עדיף.
        """
        if self.circuit_open(): return -1e9
        base = 1.0 if self.healthy else 0.0
        age = time.time() - self.last_ok
        age_bonus = 0.2 if age < 2.0 else 0.0
        inflight_penalty = 0.02*self.inflight
        ewma_penalty = 0.0 if self.ewma_ms is None else min(self.ewma_ms/2000.0, 1.0)
        shed_penalty = 1.0 if self.load_shed() else 0.0
        return base + age_bonus - inflight_penalty - ewma_penalty - shed_penalty

class HealthChecker:
    def __init__(self, backends: Dict[str, BackendState], *, interval_s: float=1.0):
        self.backends = backends
        self.interval_s = float(interval_s)
        self._stop = asyncio.Event()

    def stop(self) -> None: self._stop.set()

    async def run(self) -> None:
        sbx = SandboxRuntime(allow_hosts=["127.0.0.1","localhost"], http_tps=20.0, max_sleep_ms=500)
        while not self._stop.is_set():
            for b in self.backends.values():
                try:
                    t0 = time.perf_counter()
                    status, hdr, body = await sbx.http_get(b.host, b.port, "/health", timeout_s=1.0)
                    dt = (time.perf_counter() - t0)*1000.0
                    b.record_latency(dt)
                    if status == 200:
                        b.mark_ok()
                        metrics.inc(f"mesh.backend.{b.name}.health_ok", 1)
                    else:
                        b.mark_fail()
                        metrics.inc(f"mesh.backend.{b.name}.health_bad", 1)
                except Exception:
                    b.mark_fail()
                    metrics.inc(f"mesh.backend.{b.name}.health_err", 1)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_s)
            except asyncio.TimeoutError:
                pass