from __future__ import annotations
from typing import Any, Dict, List
from fastapi import APIRouter
from pydantic import BaseModel
from engine.research.hypothesis_lab import run_offline_experiments

router = APIRouter(prefix="/research", tags=["research"]) 

class RunIn(BaseModel):
    configs: List[Dict[str,Any]]

@router.post("/offline")
async def offline(inp: RunIn):
    # דוגמה ל‑runner: מודד p95 מדומה לפי פרמטרים
    def _runner(cfg: Dict[str,Any]) -> Dict[str,Any]:
        import random
        base = 1200.0 if cfg.get("model")=="gpt-4o-mini" else 1600.0
        jitter = random.uniform(-150, 150)
        return {"p95_ms": max(200.0, base + jitter)}
    return run_offline_experiments(inp.configs, runner=_runner)