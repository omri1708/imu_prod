# server/emergency_api.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from server.scheduler_api import _http_call
from policy.rbac import require_perm

router = APIRouter(prefix="/controlplane/emergency", tags=["emergency"])

class RollbackReq(BaseModel):
    target: str = Field("umbrella", description="umbrella|control-plane")
    release: str
    namespace: str
    revision: int = 1
    execute: bool = True

@router.post("/rollback")
def rollback(req: RollbackReq):
    require_perm("demo-user", "emergency:rollback")
    body={"user_id":"demo-user","kind":"helm.rollback",
          "params":{"release":req.release,"revision":req.revision,"namespace":req.namespace},
          "execute": req.execute}
    out=_http_call("POST","/adapters/run", body)
    return {"ok": out.get("ok",False), "cmd": out.get("cmd"), "reason": out.get("reason")}