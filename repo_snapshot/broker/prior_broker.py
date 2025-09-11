# broker/prio_broker.py (כבר קיימת ל־Back-pressure/QoS; מוודאים תמיכה גלובלית)
# -*- coding: utf-8 -*-
from __future__ import annotations
import time, threading, queue
from typing import Dict, Any, Iterator

class PriorityBroker:
    def __init__(self, qsize:int=10000):
        self._subscribers = {}
        self.qsize = qsize
        self._lock = threading.Lock()

    def publish(self, topic:str, event:Dict[str,Any]):
        with self._lock:
            subs = [q for t,q in self._subscribers.items() if t==topic]
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                # back-pressure: אם התמלא—נפיל lowest-priority או נשמור N אחרונים בלבד (פשטני כאן)
                pass

    def subscribe(self, topic:str) -> Iterator[Dict[str,Any]]:
        q = queue.Queue(maxsize=self.qsize)
        with self._lock:
            self._subscribers[topic] = q
        try:
            while True:
                yield q.get()
        finally:
            with self._lock:
                self._subscribers.pop(topic, None)

GLOBAL_BROKER = PriorityBroker()
