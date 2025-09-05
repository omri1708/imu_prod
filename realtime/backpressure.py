# realtime/backpressure.py 
# -*- coding: utf-8 -*-
import time, threading
from collections import defaultdict, deque

class GlobalTokenBucket:
    """
    דלי טוקנים גלובלי + החלקת burst (EWMA) + הוגנות בין topics.
    capacity: כמות הטוקנים המקסימלית במאגר.
    rate_tokens_per_sec: קצב מילוי טוקנים לשניה.
    """
    def __init__(self, capacity: int, rate_tokens_per_sec: float, alpha: float = 0.2):
        self.capacity = max(1, int(capacity))
        self.rate = float(rate_tokens_per_sec)
        self._tokens = float(capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()
        # החלקת עומס (EWMA) פר־topic: מודד צריכה אחרונה ל־smoothing
        self._ewma = defaultdict(float)
        self._alpha = alpha  # 0..1; גבוה=רגיש יותר לדגימות אחרונות

    def _refill(self):
        now = time.monotonic()
        dt = now - self._last
        if dt <= 0:
            return
        add = dt * self.rate
        self._tokens = min(self.capacity, self._tokens + add)
        self._last = now

    def try_consume(self, tokens: int = 1, topic: str = None) -> bool:
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                if topic is not None:
                    # מעדכן EWMA צריכה פר־topic (לשימוש הוגנות)
                    self._ewma[topic] = (1 - self._alpha) * self._ewma[topic] + self._alpha * tokens
                return True
            return False

    def budget_hint(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens

    def topic_load(self, topic: str) -> float:
        return self._ewma[topic]