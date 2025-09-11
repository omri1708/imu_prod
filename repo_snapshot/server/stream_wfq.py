# server/stream_wfq.py
from __future__ import annotations
import time, threading, queue, hashlib, json
from typing import Dict, Any, Tuple, Optional, List

class TokenBucket:
    def __init__(self, rate_qps: float, burst: int):
        self.rate = rate_qps; self.capacity = burst; self.tokens = burst
        self.last = time.time(); self.lock = threading.Lock()
    def take(self, n=1) -> bool:
        with self.lock:
            now=time.time()
            self.tokens = min(self.capacity, self.tokens + (now-self.last)*self.rate)
            self.last = now
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False

class WFQBroker:
    """
    Weighted Fair Queuing + token buckets per topic and per producer.
    """
    def __init__(self, global_rate: float = 200, global_burst: int = 100):
        self.q: Dict[str, queue.PriorityQueue] = {}        # topic -> PQ[(prio, ts, producer_id, event)]
        self.topic_tb: Dict[str, TokenBucket] = {}
        self.prod_tb: Dict[str, TokenBucket] = {}
        self.weights: Dict[str, int] = {}                  # topic -> weight
        self.global_tb = TokenBucket(global_rate, global_burst)
        self.lock = threading.Lock()

    def _pid(self, producer: str) -> str:
        return hashlib.sha256(producer.encode()).hexdigest()[:8]

    def ensure_topic(self, topic: str, rate: float = 50.0, burst: int = 100, weight: int = 1):
        with self.lock:
            self.q.setdefault(topic, queue.PriorityQueue())
            self.topic_tb.setdefault(topic, TokenBucket(rate, burst))
            self.weights[topic] = weight

    def submit(self, topic: str, producer: str, event: Dict[str,Any], priority: int = 10):
        pid = self._pid(producer)
        self.prod_tb.setdefault(pid, TokenBucket(rate_qps=25.0, burst=5))
        if not self.prod_tb[pid].take():  # per producer fairness
            return False
        if not self.topic_tb[topic].take():
            return False
        if not self.global_tb.take():
            return False
        ts=time.time()
        self.q[topic].put((priority, ts, pid, event))
        return True

    def poll(self, topic: str, max_items: int = 50) -> List[Dict[str,Any]]:
        out=[]
        q=self.q.get(topic)
        if not q:
            return out
        while not q.empty() and len(out)<max_items:
            pr,ts,pid,ev=q.get_nowait()
            out.append(ev)
        return out

BROKER = WFQBroker()