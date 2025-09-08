# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
# Pydantic v1 (תואם לדוקר)
try:
    from pydantic import BaseModel, Field
except Exception:
    from pydantic.v1 import BaseModel, Field  # type: ignore

# Orchestrator לבנייה בפועל
try:
    from program.orchestrator import ProgramOrchestrator
except Exception as e:
    ProgramOrchestrator = None
    _orch_import_err = e

router = APIRouter(prefix="/program", tags=["program"])

class Service(BaseModel):
    type: str = Field(..., description="e.g. python_web | python_app")
    name: str = Field(..., min_length=1)

class BuildSpec(BaseModel):
    name: str = Field(..., min_length=1)
    services: List[Service]

class BuildRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    spec: BuildSpec

@router.post("/build")
async def build(req: BuildRequest):
    if ProgramOrchestrator is None:
        raise HTTPException(500, f"Orchestrator not available: {_orch_import_err}")
    try:
        orch = ProgramOrchestrator()
        result = await orch.build(req.user_id, req.spec.dict())
        return {"ok": True, "build": result}
    except Exception as e:
        raise HTTPException(400, f"build failed: {e}")
