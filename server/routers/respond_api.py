from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from assurance.respond_text import GroundedResponder
from assurance.errors import RefusedNotGrounded, ResourceRequired, ValidationFailed
from fastapi import Depends
from server.deps.evidence_gate import require_citations_or_silence

router = APIRouter(
    prefix="/respond",
    tags=["respond"],
    dependencies=[Depends(require_citations_or_silence)]
)
gr = GroundedResponder("./assurance_store_text")

class Source(BaseModel):
    file: str | None = None
    url: str | None = None

class Req(BaseModel):
    prompt: str
    sources: List[Source]

@router.post("/grounded")
def respond(req: Req):
    try:
        outs = gr.respond_from_sources(req.prompt, [s.model_dump(exclude_none=True) for s in req.sources])
        return outs
    except (RefusedNotGrounded, ValidationFailed) as e:
        raise HTTPException(400, str(e))
    except ResourceRequired as e:
        return {"ok": False, "resource_required": e.what, "obtain": e.how_to_get}
