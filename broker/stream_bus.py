# broker/stream_bus.py
from __future__ import annotations
import time
from typing import Dict, List, Any, Callable
from collections import deque

class TopicBus:
    def __init__(self, per_topic_rate_limit:int=2000):
        self._sub: Dict[str,List[Callable[[dict],None]]] = {}
        self._q: Dict[str,deque] = {}
        self._last_minute: Dict[str, List[float]] = {}
        self._per_topic_rate = per_topic_rate_limit

    def publish(self, topic: str, event: dict):
        ts = time.time()
        # back-pressure: limit total published events per minute
        bucket = self._last_minute.setdefault(topic, [])
        bucket.append(ts)
        while bucket and ts - bucket[0] > 60: bucket.pop(0)
        if len(bucket) > self._per_topic_rate:
            # drop low-priority
            if event.get("priority","low") == "low":
                return
        self._q.setdefault(topic, deque()).append(event)
        for cb in self._sub.get(topic, []):
            cb(event)

    def subscribe(self, topic: str, cb):
        self._sub.setdefault(topic, []).append(cb)

bus = TopicBus()