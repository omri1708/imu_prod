# stream/broker.py
# -*- coding: utf-8 -*-
import time, threading, heapq, os
from typing import Dict, List, Tuple, Optional, Iterable, Any, Deque
from policy.user_policy import POLICIES
import asyncio, time, heapq
from collections import deque, defaultdict
from contextvars import ContextVar

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
                if left <= 0:
                    return None
                self._cv.wait(min(0.25, left))

_current_user: ContextVar[str] = ContextVar("imu_user_id", default=os.getenv("IMU_TEST_USER", "test"))

_brokers: Dict[str, "Broker"] = {}
class Broker:
    """
    תור עדיפויות פר-משתמש+נושא, עם מגבלות קצבים, תקרת burst גלובלית,
    ושירות פרסום/מנוי אסינכרוני.
    """
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.pol = POLICIES.get(user_id)
        self._pq = []  # heap of (priority, ts, seq, topic, payload)
        self._seq = 0
        self._subs: Dict[str, asyncio.Queue] = {}
        self._rate_buckets: Dict[str, Tuple[float, float]] = {}  # topic -> (allowance, last_ts)
        self._global_window = []
        self._global_window_sec = 1.0

    def _priority_of(self, topic: str) -> int:
        return (self.pol.priority_overrides or {}).get(topic, 5)

    def _rate_limit_ok(self, topic: str) -> bool:
        limit = (self.pol.topic_rate_limits or {}).get(topic)
        now = time.time()
        # per-topic token bucket
        if limit:
            allowance, last = self._rate_buckets.get(topic, (limit, now))
            elapsed = max(0.0, now - last)
            allowance = min(limit, allowance + elapsed * limit)
            if allowance < 1.0:
                # נחסום הודעה זו; caller יוכל לנסות שוב
                self._rate_buckets[topic] = (allowance, now)
                return False
            allowance -= 1.0
            self._rate_buckets[topic] = (allowance, now)
        # global burst guard
        self._global_window = [t for t in self._global_window if now - t < self._global_window_sec]
        if len(self._global_window) >= self.pol.burst_limit_global:
            return False
        self._global_window.append(now)
        return True

    async def publish(self, topic: str, payload: Any):
        if not self._rate_limit_ok(topic):
            return False
        pr = self._priority_of(topic)
        self._seq += 1
        item = (pr, time.time(), self._seq, topic, payload)
        heapq.heappush(self._pq, item)
        await self._drain()
        return True

    async def _drain(self):
        # מפיץ אל כל המנויים
        while self._pq:
            pr, ts, seq, topic, payload = heapq.heappop(self._pq)
            q = self._subs.get(topic)
            if q:
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    # אם הלקוח איטי—נפיל הודעות לוג לפני טלמטריה
                    pass

    def subscribe(self, topic: str, max_queue: int = 1000) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=max_queue)
        self._subs[topic] = q
        return q
    
def get_broker(user_id: Optional[str] = None) -> Broker:
    """החזר ברוקר עבור user_id; צור אם חסר (Lazy)."""
    uid = user_id or _current_user.get()
    b = _brokers.get(uid)
    if b is None:
        b = Broker(uid)
        _brokers[uid] = b
    return b

BROKER = get_broker()

def set_current_user(user_id: str) -> None:
    _current_user.set(user_id)


#------
class StreamBroker:
    def __init__(self, max_global_qps: float, max_global_burst: int):
        self.topics: Dict[str, Deque[Dict[str, Any]]] = defaultdict(deque)
        self.lock = threading.Lock()
        self.last_emit_ts = 0.0
        self.tokens = max_global_burst
        self.rps = max_global_qps
        self.max_burst = max_global_burst
        self.topic_qps: Dict[str, float] = {}
        self.topic_burst: Dict[str, int] = {}
        self.topic_tokens: Dict[str, float] = defaultdict(lambda: 0.0)
        self.priorities: Dict[str, int] = {}  # נמוך=0, גבוה=10

    @staticmethod
    def now_ms() -> int:
        return int(time.time()*1000)

    def ensure_topic(self, topic: str, qps: float, burst: int, priority: int = 5):
        with self.lock:
            self.topic_qps[topic] = qps
            self.topic_burst[topic] = burst
            self.priorities[topic] = priority
            self.topic_tokens[topic] = burst

    def _refill(self, dt: float):
        self.tokens = min(self.max_burst, self.tokens + self.rps*dt)
        for t in list(self.topic_tokens.keys()):
            cap = self.topic_burst.get(t, self.max_burst)
            self.topic_tokens[t] = min(cap, self.topic_tokens[t] + self.topic_qps.get(t, self.rps)*dt)

    def publish(self, topic: str, event: Dict[str, Any]):
        now = time.time()
        with self.lock:
            dt = max(0.0, now - self.last_emit_ts)
            self._refill(dt)
            self.last_emit_ts = now
            if self.topic_tokens.get(topic, 0.0) < 1.0 or self.tokens < 1.0:
                # דריסה שקטה/דחייה לפי מדיניות; כאן נשמור ונשדר כשיהיו טוקנים
                self.topics[topic].append(event)
                return False
            # צריכת טוקנים
            self.topic_tokens[topic] -= 1.0
            self.tokens -= 1.0
            self.topics[topic].append(event)
            return True

    def poll(self, topic: str, max_items: int = 100) -> List[Dict[str, Any]]:
        with self.lock:
            out = []
            dq = self.topics.get(topic)
            if not dq:
                return out
            while dq and len(out) < max_items:
                out.append(dq.popleft())
            return out