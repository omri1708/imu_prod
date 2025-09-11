from __future__ import annotations
from typing import Any, Dict
from fastapi import APIRouter
from pydantic import BaseModel

from engine.prebuild.adapter_builder import ensure_capabilities
from engine.prebuild.tool_acquisition import ensure_tools

router = APIRouter(prefix="/prebuild", tags=["prebuild"])

class PrebuildIn(BaseModel):
    spec: Any
    ctx: Dict[str,Any] = {}

@router.post("/ensure")
async def ensure(inp: PrebuildIn):
    ctx = dict(inp.ctx or {})
    ctx.setdefault("user_id", ctx.get("user") or "anon")
    m = ensure_capabilities(inp.spec, ctx)
    t = ensure_tools(inp.spec, ctx)
    return {"ok": True, "missing_built": m, "tools": t}