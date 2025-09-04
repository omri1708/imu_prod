# imu_repo/sandbox/limits.py
from __future__ import annotations
import time, asyncio
from typing import Dict, Tuple

class TokenBucket:
    def __init__(self, rate_per_sec: float, capacity: int) -> None:
        self.rate = float(rate_per_sec)
        self.capacity = int(capacity)
        self.tokens = float(capacity)
        self.updated = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, amount: float = 1.0) -> None:
        async with self._lock:
            while True:
                now = time.time()
                elapsed = max(0.0, now - self.updated)
                self.updated = now
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                if self.tokens >= amount:
                    self.tokens -= amount
                    return
                need = amount - self.tokens
                wait_s = need / max(self.rate, 1e-9)
                await asyncio.sleep(min(wait_s, 0.25))  # חתיכות קטנות כדי להיות רספונסיבי

class RateLimiter:
    """
    RateLimiter פר־מפתח (למשל (user,host)).
    """
    def __init__(self, rate_per_sec: float, burst: int) -> None:
        self.rate = float(rate_per_sec)
        self.burst = int(burst)
        self._buckets: Dict[Tuple[str, str], TokenBucket] = {}

    def bucket(self, user_id: str, host: str) -> TokenBucket:
        key = (str(user_id), str(host))
        b = self._buckets.get(key)
        if b is None:
            b = TokenBucket(self.rate, self.burst)
            self._buckets[key] = b
        return b

    async def acquire(self, user_id: str, host: str, amount: float = 1.0) -> None:
        await self.bucket(user_id, host).acquire(amount)