from __future__ import annotations
import time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import median
from typing import Any, Dict, List
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/research", tags=["research"])

class OnlineIn(BaseModel):
    urls: List[str] = Field(..., min_items=1)
    total_requests: int = Field(50, ge=1, le=2000)
    concurrency: int = Field(8, ge=1, le=128)
    timeout_s: float = Field(5.0, ge=0.1, le=60.0)

def _get(url: str, timeout_s: float) -> float:
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=timeout_s) as r:
        _ = r.read(64)  # מחמם בלבד
    return (time.time() - t0) * 1000.0

@router.post("/online")
async def online(inp: OnlineIn):
    latencies: List[float] = []
    with ThreadPoolExecutor(max_workers=inp.concurrency) as ex:
        futs = []
        n = max(1, inp.total_requests // max(1, len(inp.urls)))
        for _ in range(n):
            for url in inp.urls:
                futs.append(ex.submit(_get, url, inp.timeout_s))
        for f in as_completed(futs):
            try:
                latencies.append(f.result())
            except Exception:
                pass
    latencies.sort()
    p95 = latencies[int(0.95*len(latencies))-1] if latencies else None
    return {"ok": True, "n": len(latencies), "p95_ms": p95, "avg_ms": (sum(latencies)/len(latencies) if latencies else None)}
