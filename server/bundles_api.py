# server/bundles_api.py
# Create/List/Download/Verify bundles: ZIP + Envelope (Ed25519) על המטאדאטה.
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path
import os, json, time, hashlib

from provenance.keyring import Keyring
from provenance.envelope import sign_cas_record, verify_envelope, Envelope, Signature
from policy.rbac import require_perm

router = APIRouter(prefix="/bundles", tags=["bundles"])
ROOT = Path(".")
BUNDLES_DIR = ROOT/".imu"/"bundles"
BUNDLES_DIR.mkdir(parents=True, exist_ok=True)
KR = Keyring(".imu/keys")

INCLUDE_DIRS = [
    "security/policy_rules.yaml",
    ".imu/keys/pub",
    ".imu/provenance",
    ".imu/artifacts",
    "cas",
]

def _zip_bytes() -> bytes:
    buf = BytesIO()
    with ZipFile(buf, "w", compression=ZIP_DEFLATED) as z:
        for rel in INCLUDE_DIRS:
            p = ROOT/rel
            if not p.exists(): continue
            if p.is_file():
                z.write(p, arcname=rel)
            else:
                for f in p.rglob("*"):
                    if f.is_file():
                        z.write(f, arcname=str(f.relative_to(ROOT)))
        z.writestr("bundles/when.json", json.dumps({"ts": time.time()}, ensure_ascii=False, indent=2))
    return buf.getvalue()

class CreateReq(BaseModel):
    user_id: str = "demo-user"
    name: str
    comment: str = "bundle via api"

@router.post("/create")
def create_bundle(req: CreateReq):
    require_perm(req.user_id, "bundles:create")
    data = _zip_bytes()
    digest = hashlib.sha256(data).hexdigest()
    # write zip
    zip_path = BUNDLES_DIR / f"{req.name}.zip"
    zip_path.write_bytes(data)
    # sign envelope (record = {name,digest,size,ts})
    priv = KR.load_private()
    kid  = KR.current_kid() or KR.rotate("auto").kid
    record = {"name": req.name, "digest": digest, "size": len(data), "ts": time.time(), "comment": req.comment}
    env = sign_cas_record(priv, kid, record)
    env_path = BUNDLES_DIR / f"{req.name}.envelope.json"
    env_path.write_text(json.dumps({
        "payloadType": env.payloadType,
        "payload_b64": env.payload_b64,
        "signatures": [s.__dict__ for s in env.signatures]
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "name": req.name, "zip": str(zip_path), "envelope": str(env_path), "digest": digest}

@router.get("/list")
def list_bundles():
    out=[]
    for f in BUNDLES_DIR.glob("*.zip"):
        env = f.with_suffix(".envelope.json")
        out.append({"name": f.stem, "zip": str(f), "envelope": str(env if env.exists() else "")})
    return {"ok": True, "items": out}

@router.get("/download")
def download(name: str):
    p = BUNDLES_DIR / f"{name}.zip"
    if not p.exists(): raise HTTPException(404, "bundle not found")
    return FileResponse(str(p), media_type="application/zip", filename=p.name)

class VerifyReq(BaseModel):
    name: str

@router.post("/verify")
def verify(req: VerifyReq):
    zip_path = BUNDLES_DIR / f"{req.name}.zip"
    env_path = BUNDLES_DIR / f"{req.name}.envelope.json"
    if not zip_path.exists() or not env_path.exists():
        raise HTTPException(404, "bundle or envelope not found")
    env_json = json.loads(env_path.read_text(encoding="utf-8"))
    env = Envelope(payloadType=env_json["payloadType"],
                   payload_b64=env_json["payload_b64"],
                   signatures=[Signature(**s) for s in env_json["signatures"]])
    kid = env.signatures[0].kid
    pub = KR.load_public(kid)
    ok = verify_envelope(pub, env)
    # verify digest matches envelope payload
    payload = json.loads((env.payload_b64.encode("utf-8")).decode("utf-8")) if False else json.loads(bytes.fromhex(""))
    # פענוח DSSE payload: env.payload_b64 מכיל JSON קאנוני של הרשומה
    import base64
    rec = json.loads(base64.b64decode(env.payload_b64.encode()).decode())
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    return {"ok": ok and rec.get("digest")==digest, "kid": kid, "digest": digest, "envelope_digest": rec.get("digest")}