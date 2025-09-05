# streams/broker.py
import asyncio, time
from enum import IntEnum
from collections import defaultdict, deque
from typing import Any, Dict, Deque, Tuple

class Priority(IntEnum):
    TELEMETRY=0  # חשוב
    LOGIC=1
    EVENTS=2
    LOGS=3      # פחות חשוב

class TokenBucket:
    def __init__(self, rate_per_sec:float, burst:int):
        self.rate = rate_per_sec
        self.capacity = burst
        self.tokens = burst
        self.last = time.time()
    def allow(self)->bool:
        now=time.time()
        self.tokens = min(self.capacity, self.tokens + (now-self.last)*self.rate)
        self.last=now
        if self.tokens>=1:
            self.tokens -= 1
            return True
        return False

class Broker:
    def __init__(self, default_rate_per_min:int=60, default_burst:int=20):
        self.subs: Dict[str, Dict[int, asyncio.Queue]] = defaultdict(dict) # topic->sid->queue
        self.next_sid=1
        self.queues: Dict[str, Dict[Priority, Deque]] = defaultdict(lambda: defaultdict(deque))
        self.token: Dict[str, TokenBucket] = {}
        self.rate_per_min = default_rate_per_min
        self.burst = default_burst
        self._lock = asyncio.Lock()

    def _bucket(self, topic:str)->TokenBucket:
        if topic not in self.token:
            self.token[topic] = TokenBucket(rate_per_sec=self.rate_per_min/60.0, burst=self.burst)
        return self.token[topic]

    async def publish(self, topic:str, payload:Dict[str,Any], pri:Priority=Priority.EVENTS):
        # Throttling per-topic
        if not self._bucket(topic).allow(): return
        self.queues[topic][pri].append(payload)
        await self._drain(topic)

    async def _drain(self, topic:str):
        async with self._lock:
            for pri in sorted(self.queues[topic].keys()):
                q = self.queues[topic][pri]
                while q:
                    item = q.popleft()
                    dead=[]
                    for sid,aq in self.subs[topic].items():
                        try:
                            aq.put_nowait(item)
                        except asyncio.QueueFull:
                            dead.append(sid)
                    for sid in dead:
                        self.subs[topic].pop(sid, None)

    def subscribe(self, topic:str, max_queue:int=256)->Tuple[int, asyncio.Queue]:
        q = asyncio.Queue(maxsize=max_queue)
        sid = self.next_sid; self.next_sid+=1
        self.subs[topic][sid]=q
        return sid, q

    def unsubscribe(self, topic:str, sid:int):
        self.subs[topic].pop(sid, None)

BROKER = Broker(default_rate_per_min=120, default_burst=30)