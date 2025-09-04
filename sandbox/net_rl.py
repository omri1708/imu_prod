# imu_repo/sandbox/net_rl.py
from __future__ import annotations
import time
from typing import Dict

class Bucket:
    def __init__(self, capacity: int, refill_per_s: float):
        self.cap = float(capacity)
        self.refill = float(refill_per_s)
        self.tokens = float(capacity)
        self.ts = time.time()
    def allow(self, cost: int) -> bool:
        now = time.time()
        dt = max(0.0, now - self.ts)
        self.ts = now
        self.tokens = min(self.cap, self.tokens + dt * self.refill)
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

class RateLimiter:
    def __init__(self, default_cap: int = 200_000, default_refill: float = 20_000.0):
        self.default_cap = default_cap
        self.default_refill = default_refill
        self.dom: Dict[str,Bucket] = {}
    def _host(self, url: str) -> str:
        h = url.split("://",1)[-1].split("/",1)[0].lower()
        return h
    def allow(self, url: str, cost: int) -> bool:
        h = self._host(url)
        b = self.dom.get(h)
        if not b:
            b = self.dom[h] = Bucket(self.default_cap, self.default_refill)
        return b.allow(cost)