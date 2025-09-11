# server/provenance_api.py
# FastAPI router for key management, rotation, signing CAS digests, and verification.
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from pathlib import Path
import json

from provenance.keyring import Keyring
from provenance.envelope import sign_cas_record, verify_envelope, Envelope, Signature
from cryptography.hazmat.primitives import serialization

router = APIRouter(prefix="/provenance", tags=["provenance"])
KR = Keyring(".imu/keys")

class KeyGen(BaseModel):
    comment: str = "generated"

@router.post("/keys/rotate")
def rotate(k: KeyGen):
    meta = KR.rotate(comment=k.comment)
    return {"ok": True, "current": meta.kid, "meta": meta.__dict__}

@router.get("/keys")
def list_keys():
    return {"keys": [m.__dict__ for m in KR.list()]}

class CasSign(BaseModel):
    digest: str
    kind: str = "artifact"
    meta: Dict[str, Any] = {}

@router.post("/sign/cas")
def sign_cas(req: CasSign):
    priv = KR.load_private()
    kid  = KR.current_kid()
    if not kid:
        raise HTTPException(400, "no active key")
    rec = {"digest": req.digest, "kind": req.kind, "meta": req.meta}
    env = sign_cas_record(priv, kid, rec)
    # persist envelope for audit
    out_dir = Path(".imu/provenance")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"env_{req.digest}.json"
    out_path.write_text(json.dumps({
        "payloadType": env.payloadType,
        "payload_b64": env.payload_b64,
        "signatures": [s.__dict__ for s in env.signatures]
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "envelope_path": str(out_path)}

class VerifyReq(BaseModel):
    envelope_path: str

@router.post("/verify")
def verify(req: VerifyReq):
    p = Path(req.envelope_path)
    if not p.exists():
        raise HTTPException(404, "envelope not found")
    env_json = json.loads(p.read_text(encoding="utf-8"))
    env = Envelope(payloadType=env_json["payloadType"],
                   payload_b64=env_json["payload_b64"],
                   signatures=[Signature(**s) for s in env_json["signatures"]])
    # use active public key (simple trust model; can extend with kid)
    kid = env.signatures[0].kid
    pub = KR.load_public(kid)
    ok = verify_envelope(pub, env)
    return {"ok": ok, "kid": kid}