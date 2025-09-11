# streaming/broker.py
import threading
from typing import Dict, List, Callable, Any

class StreamBroker:
    def __init__(self):
        self.subs: Dict[str, List[Callable[[Any], None]]] = {}
        self.lock = threading.Lock()

    def subscribe(self, topic: str, cb: Callable[[Any], None]):
        with self.lock:
            self.subs.setdefault(topic, []).append(cb)

    def publish(self, topic: str, msg: Any):
        with self.lock:
            cbs = list(self.subs.get(topic, []))
        for cb in cbs:
            try:
                cb(msg)
            except Exception:
                pass