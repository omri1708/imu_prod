from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from user_model.model import UserStore, UserModel

router = APIRouter(prefix="/consent", tags=["consent"])
store = UserStore("./assurance_store_users")
um = UserModel(store)

class GrantReq(BaseModel):
    user_id: str
    purpose: str = "adapters/run"
    ttl_seconds: int = 3600

@router.post("/grant")
def grant(req: GrantReq):
    um.identity_register(req.user_id, {"created_by":"api"})
    um.consent_grant(req.user_id, req.purpose, req.ttl_seconds)
    return {"ok": True}
