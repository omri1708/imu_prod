from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import asyncio
from program.orchestrator import ProgramOrchestrator
from assurance.errors import ResourceRequired, ValidationFailed

router = APIRouter(prefix="/program", tags=["program"])
orch = ProgramOrchestrator("./assurance_store_programs")

class BuildReq(BaseModel):
    user_id: str
    spec: Dict[str, Any]

@router.post("/build")
def build(req: BuildReq):
    try:
        r = asyncio.get_event_loop().run_until_complete(orch.build(req.user_id, req.spec))
        return r
    except (ValidationFailed,) as e:
        raise HTTPException(400, str(e))
    except ResourceRequired as e:
        return {"ok": False, "resource_required": e.what, "obtain": e.how_to_get}
