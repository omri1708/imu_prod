# imu_repo/dist/service_registry.py
from __future__ import annotations
import time
from typing import Dict, Any, List, Optional

class ServiceRegistry:
    def __init__(self):
        self._svcs: Dict[str, Dict[str, Any]] = {}  # name -> { inst_id -> info }

    def register(self, name: str, inst_id: str, addr: str, *, meta: Dict[str,Any] | None=None):
        self._svcs.setdefault(name, {})[inst_id] = {"addr": addr, "meta": meta or {}, "last_ok": time.time(), "health":"unknown", "rr":0}

    def instances(self, name: str) -> Dict[str,Any]:
        return self._svcs.get(name, {})

    def set_health(self, name: str, inst_id: str, status: str) -> None:
        svc = self._svcs.get(name, {})
        if inst_id in svc:
            svc[inst_id]["health"] = status
            if status == "ok":
                svc[inst_id]["last_ok"] = time.time()

    def pick(self, name: str) -> Optional[Dict[str,Any]]:
        svc = self._svcs.get(name, {})
        healthy = [v for v in svc.values() if v.get("health") == "ok"]
        if not healthy:
            return None
        # round-robin
        healthy.sort(key=lambda x: x.get("rr", 0))
        chosen = healthy[0]
        chosen["rr"] = chosen.get("rr",0) + 1
        return chosen
