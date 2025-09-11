# streams/backpressure.py
from __future__ import annotations
import time, threading, queue
from typing import Any, Dict, Tuple

class TopicPolicy:
    def __init__(self, rate_per_sec: int = 20, burst: int = 40, priority: int = 5):
        self.rate = max(1, rate_per_sec); self.burst = max(1, burst); self.priority = max(0, priority)

class BackPressureBus:
    """
    Global back-pressure with token buckets per topic and a global cap.
    Items are (priority, t, topic, payload). Lower priority value == higher priority.
    """
    def __init__(self, global_burst: int = 1000):
        self.global_burst = global_burst
        self._q = queue.PriorityQueue()
        self._tokens: Dict[str, Tuple[float, float, int]] = {}  # topic -> (last_refill, tokens, rate)
        self._policies: Dict[str, TopicPolicy] = {}
        self._lock = threading.Lock()

    def set_policy(self, topic: str, policy: TopicPolicy):
        with self._lock: self._policies[topic] = policy; self._tokens[topic] = (time.time(), policy.burst, policy.rate)

    def offer(self, topic: str, payload: Any):
        pol = self._policies.get(topic, TopicPolicy())
        self._q.put((pol.priority, time.time(), topic, payload))

    def _refill(self, topic: str):
        pol = self._policies.get(topic, TopicPolicy())
        t, tokens, rate = self._tokens.get(topic, (time.time(), pol.burst, pol.rate))
        now = time.time(); delta = now - t
        add = delta * rate
        tokens = min(pol.burst, tokens + add)
        self._tokens[topic] = (now, tokens, rate)
        return tokens

    def take(self, block=True, timeout=None):
        pr, ts, topic, payload = self._q.get(block=block, timeout=timeout)
        tokens = self._refill(topic)
        if tokens >= 1:
            # consume token
            t, _, rate = self._tokens[topic]
            self._tokens[topic] = (t, tokens-1, rate)
            return (topic, payload)
        else:
            # no tokens; requeue with small delay
            time.sleep(0.01)
            self._q.put((pr, time.time(), topic, payload))
            return None