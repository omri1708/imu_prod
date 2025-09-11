# server/routers/orchestrate_api.py
from __future__ import annotations
from typing import Any, Dict, Optional
from fastapi import APIRouter, Query, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from pathlib import Path
import asyncio, json

from engine.pipelines.orchestrator import Orchestrator, default_runners
from engine.pipelines import shims  # מיפוי ידני
from engine.orchestrator.run_store import run_context, gc_old_runs, RUNS_DIR

# אכיפה: evidence + sandbox (רק אם חיברת את ה-deps האלה, מומלץ)
from server.deps.evidence_gate import require_citations_or_silence
from server.deps.sandbox import require_sandbox_ready
from engine.orchestrator.universal_orchestrator import UniversalOrchestrator
univ = UniversalOrchestrator()

router = APIRouter(
    prefix="/orchestrate",
    tags=["orchestrate"],
    dependencies=[Depends(require_citations_or_silence), Depends(require_sandbox_ready)],
)

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
    # בסיס הקשר
    ctx = dict(inp.ctx or {})
    ctx.setdefault("user_id", ctx.get("user") or "anon")

    # ניהול ריצות: ניקוי ישן + יצירת run_id וספריה
    gc_old_runs()
    with run_context(user=ctx["user_id"]) as run:
        # העבר הקשר ריצה קדימה
        audit_dir = Path(run.path) / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        ctx["run_id"] = run.id
        ctx["run_dir"] = run.path
        ctx["audit_path"] = str(audit_dir / "pipeline.jsonl")  # ← שתף זאת עם השכבות שמתעדות פרוגרס

        # הרצה בפועל (דרך shims או ה-Orchestrator שלך)
        if kind:
            fn = _KIND_MAP.get(kind)
            if not fn:
                return JSONResponse({"ok": False, "error": f"unknown kind: {kind}"}, status_code=400)
            out = await fn(inp.spec, ctx)
        else:
            out = await univ.execute(inp.spec, workdir=ctx["run_dir"])

        if isinstance(out, dict):
            out.setdefault("run_id", run.id)
        return JSONResponse(out)

@router.get("/stream/{run_id}")
async def stream_progress(run_id: str):
    # כל ריצה זורמת מתוך קובץ הפרוגרס שלה
    audit_path = Path(RUNS_DIR) / run_id / "audit" / "pipeline.jsonl"

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
                    ev = rec.get("event") or {}
                    meta = {**ev, **rec}
                    if meta.get("run_id") == run_id or meta.get("topic") == "progress":
                        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream")
