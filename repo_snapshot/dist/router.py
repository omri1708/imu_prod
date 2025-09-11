# imu_repo/dist/router.py
from __future__ import annotations
from typing import Dict, Any, Optional, Callable, Awaitable
import asyncio

class Router:
    def __init__(self, registry):
        self.registry = registry

    async def call(self, service: str, payload: Dict[str,Any],
                   invoke: Callable[[str, Dict[str,Any]], Awaitable[Dict[str,Any]]]) -> Dict[str,Any]:
        """
        בחירה בבריא ביותר (RR), ואם נכשלים — מנסים הבאה בתור (failover).
        """
        tried = set()
        last_err = None
        for _ in range(8):
            inst = self.registry.pick(service)
            if not inst: 
                raise RuntimeError("no_healthy_instances")
            addr = inst["addr"]
            if addr in tried:
                continue
            tried.add(addr)
            try:
                return await invoke(addr, payload)
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"all_instances_failed: {last_err}")