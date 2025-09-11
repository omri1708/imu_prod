# imu_repo/service_mesh/policy.py
from __future__ import annotations
from typing import Iterable, Generator
import random, math

def backoff_schedule(attempts: int=3, base_ms:int=50, max_ms:int=800, jitter: float=0.2) -> Generator[int, None, None]:
    """
    אקספוננציאלי עם jitter. לדוגמה: 50ms, 100ms, 200ms ...
    """
    cur = float(base_ms)
    for i in range(attempts):
        j = 1.0 + random.uniform(-jitter, jitter)
        yield int(min(cur*j, max_ms))
        cur *= 2.0