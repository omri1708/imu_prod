# imu_repo/grounded/audit.py
from __future__ import annotations
import os, json, time, hashlib
from typing import Dict, Any

class AuditLog:
    """Append-only audit log with content hash for tamper evidence."""

    def __init__(self, path:str):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path,"w"): pass

    def append(self, kind:str, detail:Dict[str,Any]) -> str:
        ts=time.time()
        entry={"ts":ts,"kind":kind,"detail":detail}
        blob=json.dumps(entry,sort_keys=True).encode()
        h=hashlib.sha256(blob).hexdigest()
        entry["hash"]=h
        with open(self.path,"a",encoding="utf-8") as f:
            f.write(json.dumps(entry,ensure_ascii=False)+"\n")
        return h

    def tail(self,n:int=10) -> list[Dict[str,Any]]:
        """Return last n entries."""
        with open(self.path,encoding="utf-8") as f:
            lines=f.readlines()[-n:]
        return [json.loads(x) for x in lines]
