# -*- coding: utf-8 -*-
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from assurance.errors import ResourceRequired, ValidationFailed, RefusedNotGrounded
from integration.adapter_wrap import run_adapter_with_assurance

router = APIRouter(prefix="/adapters/secure", tags=["adapters-secure"])

class RunReq(BaseModel):
    user_id: str
    kind: str
    params: Dict[str, Any] = {}
    execute: bool = False

@router.post("/run")
async def adapters_secure_run(req: RunReq):
    try:
        res = await run_adapter_with_assurance(req.user_id, req.kind, req.params, req.execute)
        return {"ok": True, **res}
    except ResourceRequired as e:
        return {"ok": False, "resource_required": e.what, "obtain": e.how_to_get}
    except (ValidationFailed, RefusedNotGrounded) as e:
        raise HTTPException(400, str(e))
