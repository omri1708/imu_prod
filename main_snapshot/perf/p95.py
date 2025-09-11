from __future__ import annotations
from typing import List
from collections import deque
import math

class P95Tracker:
    """
    חלון מתגלגל של מדידות ומדידת אחוזון 95.
    """
    def __init__(self, *, window:int = 1000):
        self.window = int(window)
        self._buf: deque = deque(maxlen=self.window)

    def add(self, value_ms: float) -> None:
        self._buf.append(float(value_ms))

    def count(self) -> int:
        return len(self._buf)

    def p95(self) -> float:
        if not self._buf:
            return 0.0
        arr: List[float] = sorted(self._buf)
        idx = int(math.ceil(0.95 * len(arr))) - 1
        idx = max(0, min(idx, len(arr) - 1))
        return arr[idx]