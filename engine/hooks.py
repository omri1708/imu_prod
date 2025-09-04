# imu_repo/engine/hooks.py
from __future__ import annotations
import asyncio, time
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class ThrottleConfig:
    capacity: int = 8            # מספר מקסימלי של בקשות בו-זמני
    refill_per_sec: float = 8.0  # קצב חידוש "טוקנים" לשנייה
    max_queue: int = 1024        # תור המתנה מקסימלי

class AsyncThrottle:
    """
    מצערת רכה: שילוב של semaphore (קונקרנציה) ו-token-bucket (קצב).
    התאמות 'חיות' לפי מדדים (p95/error/gate_denied)
    """
    def __init__(self, cfg: Optional[ThrottleConfig]=None):
        self.cfg = cfg or ThrottleConfig()
        self._sem = asyncio.Semaphore(self.cfg.capacity)
        self._capacity = float(self.cfg.capacity)
        self._tokens = float(self.cfg.capacity)
        self._last_refill = time.monotonic()
        self._in_use = 0
        self.max_in_use = 0
        self.enqueued = 0

    def advise_from_stats(self, stats: Dict[str,Any]) -> None:
        """
        התאמת קיבולת/קצב לפי מדדים: אם p95 גבוה או error/gate_denied עולים → מצמצם עומס.
        """
        lat = (stats.get("latency") or {})
        p95 = float(lat.get("p95_ms") or 0.0)
        err = float(stats.get("error_rate", 0.0))
        gate = float(stats.get("gate_denied_rate", 0.0))

        # בסיס: capacity יעד
        target = self.cfg.capacity
        if p95 > 120.0 or err > 0.05:
            target = max(1, int(0.25 * self.cfg.capacity))
        elif p95 > 90.0 or err > 0.02 or gate > 0.02:
            target = max(1, int(0.5 * self.cfg.capacity))

        # עדכון semaphore אם צריך (רק מצמצמים/מרחיבים ע"י החלפת מופע)
        if target != int(self._capacity):
            # בנה semaphore חדש עם קיבולת יעד; נאפס in_use למדידה בלבד
            self._sem = asyncio.Semaphore(target)
            self._capacity = float(target)

        # קצב חידוש: פרופורציונלי לקיבולת
        self.cfg.refill_per_sec = max(1.0, float(target))

    def _refill(self) -> None:
        now = time.monotonic()
        dt = max(0.0, now - self._last_refill)
        self._last_refill = now
        self._tokens = min(self._capacity, self._tokens + dt * self.cfg.refill_per_sec)

    async def _take_token(self) -> None:
        while True:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            await asyncio.sleep(0.001)

    async def acquire(self, *, timeout: Optional[float]=None) -> None:
        self.enqueued += 1
        try:
            await asyncio.wait_for(self._sem.acquire(), timeout=timeout)
        finally:
            self.enqueued -= 1
        await self._take_token()
        self._in_use += 1
        self.max_in_use = max(self.max_in_use, self._in_use)

    def release(self) -> None:
        self._in_use = max(0, self._in_use - 1)
        self._sem.release()

    def slot(self, *, timeout: Optional[float]=None):
        """
        הקשר נוח:  async with throttle.slot(): ...
        """
        throttle = self
        class _Ctx:
            async def __aenter__(self_inner):
                await throttle.acquire(timeout=timeout)
                return throttle
            async def __aexit__(self_inner, exc_type, exc, tb):
                throttle.release()
        return _Ctx()