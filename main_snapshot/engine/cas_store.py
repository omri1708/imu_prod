# imu_repo/engine/cas_store.py
from __future__ import annotations
import os, json, hashlib, tempfile, shutil
from typing import Any

def _audit_root() -> str:
    return os.environ.get("IMU_AUDIT_DIR", os.path.abspath(".imu_audit"))

def _cas_root() -> str:
    return os.path.join(_audit_root(), "cas")

def _ensure_dirs() -> None:
    os.makedirs(_cas_root(), exist_ok=True)

def _sha256(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def put_bytes(data: bytes) -> str:
    """
    שומר נתונים לפי sha256, מחזיר את ה־hash (hex).
    """
    _ensure_dirs()
    hx = _sha256(data)
    path = os.path.join(_cas_root(), hx)
    if not os.path.exists(path):
        tmp = tempfile.NamedTemporaryFile(delete=False, dir=_cas_root())
        try:
            tmp.write(data)
            tmp.flush()
        finally:
            tmp.close()
        os.replace(tmp.name, path)
    return hx

def put_json(obj: Any, *, ensure_ascii: bool = False) -> str:
    """
    שומר JSON בקאנוניקליות (sorted keys, compact) לתוך CAS, מחזיר hash.
    """
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=ensure_ascii).encode("utf-8")
    return put_bytes(data)

def get_bytes(hash_hex: str) -> bytes:
    path = os.path.join(_cas_root(), hash_hex)
    with open(path, "rb") as f:
        return f.read()

def get_json(hash_hex: str) -> Any:
    data = get_bytes(hash_hex)
    return json.loads(data.decode("utf-8"))