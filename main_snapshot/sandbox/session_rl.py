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

class SessionLimiter:
    def __init__(self, cap:int=300_000, refill:float=30_000.0):
        self.by_user: Dict[str, Bucket] = {}
        self.default_cap = cap; self.default_refill = refill
    def allow(self, user_id: str, cost: int) -> bool:
        b = self.by_user.get(user_id)
        if not b:
            b = self.by_user[user_id] = Bucket(self.default_cap, self.default_refill)
        return b.allow(cost)