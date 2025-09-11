# imu_repo/cas/store.py
from __future__ import annotations
import os, io, json, hashlib
from typing import Dict, Any, Optional, Tuple

def _root() -> str:
    d = os.environ.get("IMU_CAS_DIR") or ".cas"
    os.makedirs(d, exist_ok=True)
    return d

def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def put_bytes(b: bytes, *, media_type: str="application/octet-stream") -> Dict[str,Any]:
    h = _hash_bytes(b)
    r = _root()
    sub = os.path.join(r, h[:2], h[2:4])
    os.makedirs(sub, exist_ok=True)
    p = os.path.join(sub, h)
    if not os.path.exists(p):
        with open(p, "wb") as f:
            f.write(b)
    meta = {"media_type": media_type, "sha256": h, "size": len(b)}
    with open(p + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return {"ok": True, "sha256": h, "path": p, "media_type": media_type, "size": len(b)}

def put_json(obj: Dict[str,Any], *, media_type: str="application/json") -> Dict[str,Any]:
    b = json.dumps(obj, ensure_ascii=False, separators=(",",":")).encode("utf-8")
    return put_bytes(b, media_type=media_type)

def get(sha256: str) -> Optional[bytes]:
    r = _root()
    p = os.path.join(r, sha256[:2], sha256[2:4], sha256)
    if not os.path.exists(p): return None
    with open(p, "rb") as f:
        return f.read()

def stat(sha256: str) -> Optional[Dict[str,Any]]:
    r = _root()
    p = os.path.join(r, sha256[:2], sha256[2:4], sha256 + ".json")
    if not os.path.exists(p): return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None