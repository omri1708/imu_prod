# server/unified_archive_api.py
from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path
import json, hashlib, time, shutil, base64

from provenance.keyring import Keyring
from provenance.envelope import sign_cas_record, verify_envelope, Envelope, Signature
from server.stream_wfq import BROKER

router = APIRouter(prefix="/unified", tags=["unified-archive"])
ROOT = Path(".")
OUT_DIR = ROOT/".imu/unified"
OUT_DIR.mkdir(parents=True, exist_ok=True)
KR = Keyring(".imu/keys")

INCLUDE = [
    "security/policy_rules.yaml",
    ".imu/keys/pub",
    ".imu/provenance",
    ".imu/artifacts",
    ".imu/runbook/history",
    "cas",
]

def _zip_all() -> bytes:
    buf = BytesIO()
    with ZipFile(buf, "w", compression=ZIP_DEFLATED) as z:
        for rel in INCLUDE:
            p=ROOT/rel
            if not p.exists(): continue
            if p.is_file(): z.write(p, arcname=rel)
            else:
                for f in p.rglob("*"):
                    if f.is_file(): z.write(f, arcname=str(f.relative_to(ROOT)))
        z.writestr("unified/snapshot.json", json.dumps({"ts": time.time()}, ensure_ascii=False, indent=2))
    return buf.getvalue()

@router.get("/export_signed")
def export_signed(name: str="unified"):
    data=_zip_all()
    digest=hashlib.sha256(data).hexdigest()
    # חתימה DSSE על רשומת מטא
    priv = KR.load_private()
    kid = KR.current_kid() or KR.rotate("auto").kid
    record = {"digest": digest, "size": len(data), "ts": time.time(), "name": name}
    env = sign_cas_record(priv, kid, record)
    # שמירה לדיסק
    zip_path = OUT_DIR/f"{name}.zip"; zip_path.write_bytes(data)
    env_path = OUT_DIR/f"{name}.envelope.json"
    env_path.write_text(json.dumps({
        "payloadType": env.payloadType,
        "payload_b64": env.payload_b64,
        "signatures": [s.__dict__ for s in env.signatures]
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    BROKER.ensure_topic("timeline", rate=50.0, burst=200, weight=2)
    BROKER.submit("timeline","unified",{"type":"event","ts":time.time(),"note":f"unified.export {name}"},priority=3)
    return StreamingResponse(BytesIO(data), media_type="application/zip",
                             headers={"Content-Disposition": f"attachment; filename={name}.zip",
                                      "X-IMU-Digest": digest,
                                      "X-IMU-Envelope": str(env_path)})

@router.post("/import_signed")
async def import_signed(zip_file: UploadFile = File(...), envelope_json: UploadFile = File(...)):
    data = await zip_file.read()
    env = json.loads((await envelope_json.read()).decode("utf-8"))
    digest = hashlib.sha256(data).hexdigest()
    # אימות חתימה
    kid = env["signatures"][0]["kid"]
    pub = KR.load_public(kid)
    envelope = Envelope(payloadType=env["payloadType"],
                        payload_b64=env["payload_b64"],
                        signatures=[Signature(**s) for s in env["signatures"]])
    ok = verify_envelope(pub, envelope)
    if not ok: raise HTTPException(400, "signature invalid")
    # בדיקת digest תואם
    record=json.loads(base64.b64decode(envelope.payload_b64.encode()).decode())
    if record.get("digest") != digest:
        raise HTTPException(400, "digest mismatch")
    # חלץ לתוך .imu/imports/<name-timestamp>/
    target = OUT_DIR/("import_"+str(int(time.time())))
    target.mkdir(parents=True, exist_ok=True)
    with ZipFile(BytesIO(data), "r") as z:
        z.extractall(target)
    BROKER.ensure_topic("timeline", rate=50.0, burst=200, weight=2)
    BROKER.submit("timeline","unified",{"type":"event","ts":time.time(),"note":f"unified.import -> {str(target)}"},priority=4)
    return {"ok": True, "dir": str(target)}