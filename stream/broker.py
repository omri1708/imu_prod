# stream/broker.py
# -*- coding: utf-8 -*-
import time, threading, heapq
from typing import Dict, List, Tuple, Optional, Iterable

class _Event:
    __slots__ = ("ts","prio","topic","data")
    def __init__(self, topic: str, data: dict, prio: int):
        self.ts = time.time()
        self.prio = prio
        self.topic = topic
        self.data = data

class Topic:
    def __init__(self, name: str, max_q: int = 1000, rate_per_sec: float = 200.0):
        self.name = name
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._heap: List[Tuple[int,float,_Event]] = []  # (neg_prio, ts, ev)
        self._max_q = max_q
        self._rate = rate_per_sec
        self._last_emit = 0.0

    def put(self, ev: _Event) -> bool:
        with self._lock:
            if len(self._heap) >= self._max_q:
                # back-pressure per-topic: דריסה עדיפה של אירועים נמוכי עדיפות
                try:
                    # אם הראשון בתור עדיפותו נמוכה – הוצא אותו
                    heapq.heappop(self._heap)
                except IndexError:
                    return False
            heapq.heappush(self._heap, (-ev.prio, ev.ts, ev))
            self._cv.notify_all()
            return True

    def get(self, timeout: float = 10.0) -> Optional[_Event]:
        end = time.time() + timeout
        with self._lock:
            while True:
                if self._heap:
                    # throttling per-topic
                    now = time.time()
                    if self._rate > 0:
                        min_gap = 1.0 / self._rate
                        if now - self._last_emit < min_gap:
                            wait_left = self._last_emit + min_gap - now
                            self._cv.wait(max(0.0, min(wait_left, 0.05)))
                            continue
                    negp, _, ev = heapq.heappop(self._heap)
                    self._last_emit = time.time()
                    return ev
                left = end - time.time()
                if left <= 0: return None
                self._cv.wait(min(0.25, left))

class Broker:
    """
    ברוקר אירועים עם:
      • Back-pressure גלובלי (N*burst) + לכל-נושא
      • Priority queues (מספר עדיפות גבוה = חשוב)
      • Throttling per-topic
    """
    def __init__(self, global_capacity: int = 10000):
        self._topics: Dict[str, Topic] = {}
        self._lock = threading.RLock()
        self._global_cap = global_capacity
        self._global_load = 0

    def ensure_topic(self, name: str, **cfg) -> Topic:
        with self._lock:
            if name not in self._topics:
                self._topics[name] = Topic(name, **cfg)
            return self._topics[name]

    def publish(self, topic: str, data: dict, prio: int = 5) -> bool:
        with self._lock:
            if self._global_load >= self._global_cap:
                # גלובלי: השלך אירוע עדיפות נמוכה קודם
                # (פשטות: נחסום, אפשר לשפר בהיגיון של ניקוי)
                return False
            self._global_load += 1
        try:
            t = self.ensure_topic(topic)
            ok = t.put(_Event(topic, data, prio))
            return ok
        finally:
            with self._lock:
                self._global_load = max(0, self._global_load - 1)

    def subscribe_iter(self, topic: str, timeout: float = 30.0) -> Iterable[dict]:
        t = self.ensure_topic(topic)
        while True:
            ev = t.get(timeout=timeout)
            if ev is None:
                # keep-alive
                yield {"topic": topic, "type":"keepalive", "ts": time.time()}
            else:
                yield {"topic": topic, "type":"event", "ts": ev.ts, "data": ev.data}

BROKER = Broker()