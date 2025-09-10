# server/orchestrate_api.py
from __future__ import annotations
from typing import Any, Dict, Optional
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from engine.pipelines.orchestrator import Orchestrator, default_runners
from engine.pipelines import shims  # נשתמש במיפוי ידני
from fastapi import Response
from fastapi.responses import StreamingResponse
import asyncio, json
from pathlib import Path

router = APIRouter(prefix="/orchestrate", tags=["orchestrate"])
_orch = Orchestrator(default_runners())

_KIND_MAP = {
    "vm": shims.run_vm,
    "events": shims.run_events_spec,
    "ab_explore": shims.run_ab_explore,
    "buildspec": shims.run_buildspec,
    "buildspec_multi": shims.run_buildspec_multi,
}

class OrchestrateIn(BaseModel):
    spec: Any
    ctx: Dict[str, Any] = {}

@router.post("/run")
async def orchestrate(inp: OrchestrateIn, kind: Optional[str] = Query(None)):
    ctx = dict(inp.ctx or {})
    ctx.setdefault("user_id", ctx.get("user") or "anon")
    if kind:
        fn = _KIND_MAP.get(kind)
        if not fn:
            return JSONResponse({"ok": False, "error": f"unknown kind: {kind}"}, status_code=400)
        out = await fn(inp.spec, ctx)
    else:
        out = await _orch.run_any(inp.spec, ctx)
    return JSONResponse(out)


@router.get("/stream/{run_id}")
async def stream_progress(run_id: str):
    audit_path = Path("var/audit/pipeline.jsonl")

    async def gen():
        pos = 0
        while True:
            if not audit_path.exists():
                await asyncio.sleep(0.5)
                continue
            with audit_path.open("r", encoding="utf-8") as f:
                f.seek(pos)
                for line in f:
                    pos = f.tell()
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    # מסננים אירועי progress / timeline לאותו run_id
                    ev = rec.get("event") or {}
                    meta = {**ev, **rec}
                    if meta.get("run_id") == run_id or meta.get("topic") == "progress":
                        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream")