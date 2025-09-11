# imu_repo/sandbox/net_class_rl.py
from __future__ import annotations
import time
from typing import Dict
from grounded.trust import classify_source

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

class ClassRateLimiter:
    """
    קצב שונה לפי class:
      - official: קצבים נדיבים
      - scholarly: נדיב
      - news: בינוני
      - wiki/user: שמרני יותר
      - internal: נדיב מאוד (שלנו)
    """
    DEFAULTS = {
        "official":  (400_000, 40_000.0),
        "scholarly": (300_000, 30_000.0),
        "news":      (200_000, 20_000.0),
        "wiki":      (120_000, 12_000.0),
        "user":      (80_000,   8_000.0),
        "internal":  (500_000, 50_000.0),
    }
    def __init__(self):
        self.by_class: Dict[str, Bucket] = {}
    def _bucket_for(self, cls: str) -> Bucket:
        b = self.by_class.get(cls)
        if b: return b
        cap, refill = self.DEFAULTS.get(cls, (80_000, 8_000.0))
        b = Bucket(cap, refill)
        self.by_class[cls] = b
        return b
    def allow(self, url: str, cost: int) -> bool:
        cls = classify_source(url)
        return self._bucket_for(cls).allow(cost)