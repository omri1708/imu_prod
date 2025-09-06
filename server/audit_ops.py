# server/audit_ops.py
from __future__ import annotations
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from typing import Dict, Any
from pathlib import Path
import json, time, hashlib, base64

from provenance.keyring import Keyring
from provenance.envelope import sign_cas_record, Envelope, Signature

AUDIT_DIR = Path(".imu/audit/ops")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
KR = Keyring(".imu/keys")

def record_op(actor: str, action: str, payload: Dict[str,Any]) -> str:
    """חתימת DSSE על רשומת פעולה + שמירה בדיסק, החזרת נתיב המעטפה."""
    # מטא
    rec = {
        "actor": actor,
        "action": action,
        "ts": time.time(),
        "payload": payload
    }
    priv = KR.load_private()
    kid  = KR.current_kid() or KR.rotate("auto").kid
    env  = sign_cas_record(priv, kid, rec)
    envp = AUDIT_DIR / f"{int(time.time())}_{hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()[:16]}.json"
    envp.write_text(json.dumps({
        "payloadType": env.payloadType,
        "payload_b64": env.payload_b64,
        "signatures": [s.__dict__ for s in env.signatures]
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(envp)

class AuditMiddleware(BaseHTTPMiddleware):
    """
    חותם DSSE לכל קריאת API (שומר רק מטא וקטע מה-body לצמצום).
    הוסף ל-APP: APP.add_middleware(AuditMiddleware)
    """
    async def dispatch(self, request: Request, call_next):
        t0 = time.time()
        body_bytes = await request.body()
        try:
            body_json = json.loads(body_bytes.decode() or "{}")
        except Exception:
            body_json = {"_raw": (body_bytes[:256].decode(errors="ignore") if body_bytes else "")}
        resp: Response = await call_next(request)
        meta = {
            "path": request.url.path,
            "method": request.method,
            "status": resp.status_code,
            "elapsed_ms": int((time.time()-t0)*1000),
        }
        try:
            record_op(actor=request.headers.get("X-IMU-User","system"),
                      action=f"{request.method}:{request.url.path}",
                      payload={"req": body_json, "meta": meta})
        except Exception:
            pass
        return resp