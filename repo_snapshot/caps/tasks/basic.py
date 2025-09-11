# imu_repo/caps/tasks/basic.py
from __future__ import annotations
from typing import Dict, Any
import asyncio

from runtime.async_sandbox import SandboxRuntime
from runtime.metrics import metrics, atimer

def run_task(task: str, args: Dict[str,Any]) -> Dict[str,Any]:
    """
    משימות נתמכות: sum, sleep_ms, http_local
    """
    if task == "sum":
        a = float(args.get("a",0)); b = float(args.get("b",0))
        with_timer = args.get("_timer_key","worker.exec.sum")
        return _sum(a,b, with_timer)
    if task == "sleep_ms":
        ms = int(args.get("ms", 10))
        return asyncio.run(_sleep(ms))
    if task == "http_local":
        host = args.get("host","127.0.0.1")
        port = int(args.get("port", 80))
        path = args.get("path","/")
        return asyncio.run(_http_local(host, port, path))
    raise RuntimeError(f"unknown_task:{task}")

def _sum(a: float, b: float, timer_key: str) -> Dict[str,Any]:
    import time
    t0 = time.perf_counter()
    out = a + b
    dt = (time.perf_counter() - t0)*1000.0
    metrics.record_latency_ms(timer_key, dt)
    return {"ok": True, "result": out}

async def _sleep(ms: int) -> Dict[str,Any]:
    sbx = SandboxRuntime()
    async with atimer("worker.exec.sleep"):
        await sbx.sleep_ms(ms)
    return {"ok": True, "slept_ms": ms}

async def _http_local(host: str, port: int, path: str) -> Dict[str,Any]:
    sbx = SandboxRuntime(allow_hosts=["127.0.0.1","localhost"])
    async with atimer("worker.exec.http_local"):
        status, headers, body = await sbx.http_get(host, port, path)
    return {"ok": (status==200), "status": status, "len": len(body)}