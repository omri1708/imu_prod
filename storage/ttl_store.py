# storage/ttl_store.py
import time
from typing import Dict, Any

class TTLStore:
    def __init__(self):
        self._items: Dict[str, tuple[float, Any]] = {}

    def put(self, key: str, value: Any, ttl_sec: int):
        self._items[key] = (time.time() + ttl_sec, value)

    def get(self, key: str):
        v = self._items.get(key)
        if not v:
            return None
        exp, val = v
        if time.time() > exp:
            self._items.pop(key, None)
            return None
        return val

    def purge(self):
        now = time.time()
        for k, (exp, _) in list(self._items.items()):
            if now > exp:
                self._items.pop(k, None)

ttl = TTLStore()