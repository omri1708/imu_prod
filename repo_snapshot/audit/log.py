# audit/log.py
# -*- coding: utf-8 -*-
import hashlib, json, os, time, threading
from typing import Optional, Dict, Any


class AppendOnlyAudit:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._lock = threading.RLock()

    def append(self, obj):
        rec = dict(obj)
        rec["ts"] = time.time()
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _tail_hash(self) -> str:
        h = "0"*64
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    h = obj.get("_self_hash", h)
                except: pass
        return h

    
    @staticmethod
    def _hash_line(obj: Dict[str, Any]) -> str:
        s = json.dumps(obj, sort_keys=True, separators=(",",":"), ensure_ascii=False)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    
