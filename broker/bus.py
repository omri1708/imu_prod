# broker/bus.py (Back-pressure גלובלי + Priority Queues + Throttling per-topic)
# -*- coding: utf-8 -*-
import queue, threading, time
from typing import Dict, Any, Callable

class BusOverload(RuntimeError): pass

class EventBus:
    """
    תורים בעלי עדיפות: high (logic/telemetry), normal, low (logs).
    back-pressure: מקסימום פר נושא + נפילה/דחייה אם נפרץ.
    """
    def __init__(self, per_topic_max=1000):
        self.q_high=queue.Queue(maxsize=per_topic_max)
        self.q_norm=queue.Queue(maxsize=per_topic_max)
        self.q_low =queue.Queue(maxsize=per_topic_max)
        self.subs: Dict[str, Callable[[Dict[str,Any]],None]]={}
        self._lock=threading.RLock()
        self._stop=False
        self._worker=threading.Thread(target=self._pump, daemon=True); self._worker.start()
        self._throttle: Dict[str, float]={} # topic-> next_allowed_ts

    def subscribe(self, topic:str, handler:Callable[[Dict[str,Any]],None]):
        with self._lock: self.subs[topic]=handler

    def set_throttle(self, topic:str, per_sec:int):
        with self._lock:
            self._throttle[topic]= 0 if per_sec<=0 else (1.0/float(per_sec))

    def publish(self, topic:str, event:Dict[str,Any], priority:str="normal"):
        with self._lock:
            step=self._throttle.get(topic,0)
        if step>0:
            now=time.time()
            if not hasattr(self,"_next"): self._next={}
            nxt=self._next.get(topic,0)
            if now<nxt: 
                return  # drop כשחונקים
            self._next[topic]=now+step
        q={"high":self.q_high,"normal":self.q_norm,"low":self.q_low}.get(priority,self.q_norm)
        try:
            q.put_nowait((topic,event))
        except queue.Full:
            raise BusOverload(f"topic_overflow:{topic}")

    def _pump(self):
        while not self._stop:
            for q in (self.q_high, self.q_norm, self.q_low):
                try:
                    topic, evt = q.get(timeout=0.05)
                    h=None
                    with self._lock: h=self.subs.get(topic)
                    if h: 
                        try: h(evt)
                        except Exception: pass
                except queue.Empty:
                    pass