# broker/streams.py
import queue, threading, time
from typing import Any, Dict, Callable, List

class Broker:
    _instance = None
    @classmethod
    def singleton(cls):
        if not cls._instance: cls._instance = cls()
        return cls._instance

    def __init__(self):
        # priority: 0 = highest (control), 1 = metrics, 2 = logs
        self.queues = {0: queue.Queue(maxsize=1000), 1: queue.Queue(maxsize=5000), 2: queue.Queue(maxsize=10000)}
        self.subs: Dict[str, List[Callable[[Dict[str,Any]], None]]] = {}
        self.N_BURST = 2000  # global throttle window
        self._window = []
        self.lock = threading.Lock()

    def _admit(self):
        now = time.time()
        with self.lock:
            self._window = [t for t in self._window if now - t < 1.0]
            if len(self._window) >= self.N_BURST:
                time.sleep(0.01)
                return False
            self._window.append(now)
            return True

    def publish(self, topic: str, msg: Dict[str,Any], priority: int=1):
        if not self._admit():  # global back-pressure
            priority = min(priority+1, 2)
        q = self.queues.get(priority)
        if q.full(): return  # drop lowest-prio
        q.put({"topic": topic, "msg": msg})

    def subscribe(self, topic: str, handler: Callable[[Dict[str,Any]], None]):
        self.subs.setdefault(topic, []).append(handler)

    def pump(self):
        while True:
            for p in (0,1,2):
                try:
                    item = self.queues[p].get(timeout=0.05)
                    for h in self.subs.get(item["topic"], []):
                        h(item["msg"])
                except queue.Empty:
                    pass