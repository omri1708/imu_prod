# imu_repo/audit/signing.py
from __future__ import annotations
from typing import Dict, Any
import hmac, hashlib, json, os

def canonical(obj: Dict[str,Any]) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",",":")).encode("utf-8")

def sign(obj: Dict[str,Any], *, shared_secret: str | None=None) -> str:
    key = (shared_secret or os.environ.get("IMU_HMAC_KEY") or "imu_dev_key").encode("utf-8")
    return hmac.new(key, canonical(obj), hashlib.sha256).hexdigest()

def verify(obj: Dict[str,Any], sig: str, *, shared_secret: str | None=None) -> bool:
    try:
        return hmac.compare_digest(sign(obj, shared_secret=shared_secret), sig)
    except Exception:
        return False