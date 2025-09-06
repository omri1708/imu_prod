# server/key_admin_api.py
# Key Admin API: list/rotate/activate/export public bundle/import public key.
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
from provenance.keyring import Keyring

router = APIRouter(prefix="/keys", tags=["keys"])
KR = Keyring(".imu/keys")

class RotateReq(BaseModel):
    comment: str = "rotated via API"

@router.post("/rotate")
def rotate(req: RotateReq):
    meta = KR.rotate(comment=req.comment)
    return {"ok": True, "active": meta.kid, "meta": meta.__dict__}

@router.get("/")
def list_keys():
    items = [m.__dict__ for m in KR.list()]
    return {"ok": True, "keys": items, "active": KR.current_kid()}

class ActivateReq(BaseModel):
    kid: str = Field(..., min_length=4, description="key id to activate")

@router.post("/activate")
def activate(req: ActivateReq):
    try:
        KR.set_active(req.kid)
        return {"ok": True, "active": req.kid}
    except Exception as e:
        raise HTTPException(400, f"activate failed: {e}")

@router.get("/public_bundle")
def public_bundle():
    return {"ok": True, "bundle": KR.export_public_keys()}

class ImportPublicReq(BaseModel):
    kid: str
    pub_pem: str

@router.post("/import_public")
def import_public(req: ImportPublicReq):
    # יבוא פומבי: שומר PEM תחת pub/ ומעדכן index אם לא קיים
    pub_path = KR.root / "pub" / f"{req.kid}.pem"
    if pub_path.exists():
        raise HTTPException(400, "kid already exists")
    pub_path.write_text(req.pub_pem, encoding="utf-8")
    # רושם ברשימה כלא־פעיל
    KR.index[req.kid] = KR.index.get(req.kid) or type(KR.list()[0])(kid=req.kid, created=0.0, active=False, comment="imported")
    KR._persist()
    return {"ok": True}