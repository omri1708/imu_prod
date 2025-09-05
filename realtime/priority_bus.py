# realtime/priority_bus.py 
# -*- coding: utf-8 -*-
import asyncio, heapq, time
import contextlib
from collections import defaultdict, deque
from typing import Dict, Deque, Tuple, Any, Optional, AsyncIterator
from .backpressure import GlobalTokenBucket

class AsyncPriorityTopicBus:
    """
    תור נושאים עדיפויות עם הוגנות: Weighted Fair Scheduling בין topics,
    מחסומים פר־topic (rate-limit), ותמיכה ב-pause/resume.
    """
    def __init__(self, global_bucket: GlobalTokenBucket,
                 per_topic_rates: Dict[str, Tuple[int, float]] = None,
                 max_queue_per_topic: int = 1000):
        self.global_bucket = global_bucket
        self.per_topic_rates = per_topic_rates or {}
        self.max_queue_per_topic = max_queue_per_topic

        self._qs: Dict[str, Deque[Tuple[int, Any]]] = defaultdict(deque)
        self._paused: Dict[str, bool] = defaultdict(lambda: False)
        self._subscribers: Dict[str, Deque[asyncio.Queue]] = defaultdict(deque)
        self._per_topic_bucket: Dict[str, GlobalTokenBucket] = {}
        self._lock = asyncio.Lock()

        for topic, (cap, rate) in self.per_topic_rates.items():
            self._per_topic_bucket[topic] = GlobalTokenBucket(capacity=cap, rate_tokens_per_sec=rate)

        # מתקדם: fair loop
        self._scheduler_task: Optional[asyncio.Task] = None
        self._stop = False

    async def start(self):
        if self._scheduler_task is None:
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def stop(self):
        self._stop = True
        if self._scheduler_task:
            await asyncio.sleep(0)  # allow task to notice stop
            self._scheduler_task.cancel()
            with contextlib.suppress(Exception):
                await self._scheduler_task

    async def publish(self, topic: str, payload: Any, priority: int = 10):
        async with self._lock:
            q = self._qs[topic]
            if len(q) >= self.max_queue_per_topic:
                # דרופ חכם: מדרגים לפי עדיפות (נמוכה נזרקת קודם)
                worst_i = None
                worst_pri = -1
                for i, (p, _) in enumerate(q):
                    if p > worst_pri:
                        worst_pri, worst_i = p, i
                if worst_i is not None and worst_pri > priority:
                    q.remove(q[worst_i])
                else:
                    return  # מפוצץ → לא מכניסים
            q.append((priority, payload))

    async def subscribe(self, topic: str) -> AsyncIterator[Any]:
        q = asyncio.Queue(maxsize=100)
        self._subscribers[topic].append(q)
        try:
            while True:
                item = await q.get()
                yield item
        finally:
            self._subscribers[topic].remove(q)

    def pause(self, topic: str):
        self._paused[topic] = True

    def resume(self, topic: str):
        self._paused[topic] = False

    async def _scheduler_loop(self):
        """
        בוחר topic “צודק” לפי עומס (EWMA) ועדיפויות בפועל.
        """
        while not self._stop:
            made_progress = False
            topics = list(self._qs.keys())
            # שקלול: פחות צרך לאחרונה → מקבל עדיפות (fair-share)
            topics.sort(key=lambda t: self.global_bucket.topic_load(t))
            for topic in topics:
                if self._paused[topic]:
                    continue
                q = self._qs[topic]
                if not q:
                    continue
                # בדיקת תקציב פר־topic + גלובלי
                tb = self._per_topic_bucket.get(topic)
                if tb and (not tb.try_consume(1, topic=topic)):
                    continue
                if not self.global_bucket.try_consume(1, topic=topic):
                    continue
                # שולף לפי עדיפות (מסודר ידנית)
                # נעשה pass 1 למצוא מינימום priority (עדיפות גבוהה = ערך קטן)
                best_i = None
                best_pri = 10**9
                for i, (pri, _) in enumerate(q):
                    if pri < best_pri:
                        best_pri, best_i = pri, i
                pri, payload = q[best_i]
                del q[best_i]
                # הפצה לכל המנויים
                for sub_q in self._subscribers[topic]:
                    try:
                        sub_q.put_nowait(payload)
                    except asyncio.QueueFull:
                        # back-pressure לצד המקבל: לא נדחף בכוח
                        pass
                made_progress = True
            if not made_progress:
                await asyncio.sleep(0.002)