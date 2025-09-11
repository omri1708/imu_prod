# realtime/qos_broker.py
# -*- coding: utf-8 -*-
import time, threading, queue
from typing import Dict, Any
from contracts.errors import RateLimitExceeded
from realtime.ws_broker import start_ws_broker as _start, publish as _raw_publish

class TokenBucket:
    def __init__(self, rate_per_sec: float, burst: int):
        self.rate = rate_per_sec
        self.burst = burst
        self.tokens = burst
        self.last = time.time()
        self.lock = threading.Lock()

    def take(self, n=1) -> bool:
        with self.lock:
            now = time.time()
            elapsed = now - self.last
            self.last = now
            self.tokens = min(self.burst, self.tokens + elapsed*self.rate)
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False

class QoSBroker:
    """תור עדיפות + TokenBucket גלובלי ולכל topic. משלוח דרך ws_broker."""
    def __init__(self, global_rate=200, global_burst=400, per_topic_rate=50, per_topic_burst=100, max_queue=10000):
        self.global_bucket = TokenBucket(global_rate, global_burst)
        self.topic_buckets: Dict[str, TokenBucket] = {}
        self.q = queue.PriorityQueue(maxsize=max_queue)
        self.alive = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    def _topic(self, t: str) -> TokenBucket:
        if t not in self.topic_buckets:
            self.topic_buckets[t] = TokenBucket(50, 100)
        return self.topic_buckets[t]
    def publish(self, topic: str, payload: Dict[str,Any], priority: int = 10):
        if not self.global_bucket.take():
            raise RateLimitExceeded("global", "token_bucket")
        if not self._topic(topic).take():
            raise RateLimitExceeded(topic, "token_bucket")
        try:
            self.q.put_nowait((priority, time.time(), topic, payload))
        except queue.Full:
            raise RateLimitExceeded("queue", "max_queue")
    def _run(self):
        while self.alive:
            try:
                pr, _, t, p = self.q.get(timeout=0.1)
                _raw_publish(t, p)
            except queue.Empty:
                pass

_qos = None
def start(host="127.0.0.1", port=8766, **qos):
    global _qos
    _start(host, port)
    _qos = QoSBroker(**qos)
    return _qos

def publish(topic: str, payload: Dict[str,Any], priority: int = 10):
    if _qos is None: start()
    _qos.publish(topic, payload, priority)