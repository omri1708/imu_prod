# audit/audit_log.py
from __future__ import annotations
import time, json, hashlib
from typing import List, Dict, Any

class AuditLog:
    def __init__(self):
        self._entries: List[Dict[str,Any]] = []
        self._root_hash = "0"*64

    def append(self, event: Dict[str,Any]) -> str:
        event["ts"] = time.time()
        self._entries.append(event)
        h = hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest()
        self._root_hash = hashlib.sha256((self._root_hash + h).encode()).hexdigest()
        return h

    def export(self) -> Dict[str,Any]:
        return {"root": self._root_hash, "entries": self._entries}

audit = AuditLog()