# imu_repo/dist/health.py
from __future__ import annotations
import asyncio, random
from typing import Callable, Awaitable, Dict

async def periodic_healthcheck(instances: Dict[str, Dict], *, probe: Callable[[Dict], Awaitable[bool]], interval_s: float = 0.5):
    while True:
        for info in list(instances.values()):
            try:
                ok = await probe(info)
                info["health"] = "ok" if ok else "bad"
            except:
                info["health"] = "bad"
        await asyncio.sleep(interval_s)