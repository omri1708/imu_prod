# imu_repo/audit/cas.py
from __future__ import annotations
from typing import Dict, Any, Tuple
import os, json, hashlib, time

CAS_ROOT = "/mnt/data/imu_repo/cas"

def _ensure(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def put_bytes(b: bytes, *, meta: Dict[str,Any] | None=None) -> Dict[str,Any]:
    _ensure(CAS_ROOT)
    h = hashlib.sha256(b).hexdigest()
    sub = os.path.join(CAS_ROOT, h[:2])
    _ensure(sub)
    blob = os.path.join(sub, f"{h}.bin")
    idx  = os.path.join(sub, f"{h}.json")
    if not os.path.exists(blob):
        with open(blob, "wb") as f: f.write(b)
    rec = {
        "sha256": h,
        "size": len(b),
        "ts": time.time(),
        "meta": meta or {}
    }
    with open(idx, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    return rec

def put_json(obj: Dict[str,Any], *, meta: Dict[str,Any] | None=None) -> Dict[str,Any]:
    b = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return put_bytes(b, meta=meta)

def get_path(sha256: str) -> str:
    return os.path.join(CAS_ROOT, sha256[:2], f"{sha256}.bin")
