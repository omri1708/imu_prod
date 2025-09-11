# imu_repo/capabilities/db_memory.py
from __future__ import annotations
from typing import Dict, Any
from grounded.claims import current

class MemoryDB:
    def __init__(self) -> None:
        self._kv: Dict[str,str] = {}

    def set(self, k: str, v: str) -> None:
        self._kv[k] = v

    def get(self, k: str) -> str:
        if k not in self._kv: 
            raise KeyError(k)
        return self._kv[k]

DB = MemoryDB()

async def db_get_text(spec: Dict[str,Any]) -> str:
    """
    spec = {"key":"k"}
    """
    k = str(spec["key"])
    v = DB.get(k)
    current().add_evidence("db_memory", {
        "source_url": f"mem://db/{k}",
        "trust": 0.85,
        "ttl_s": 600,
        "payload": {"key": k, "len": len(v)}
    })
    return v