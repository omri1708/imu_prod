# evidence/cas.py (Content-Addressable Store + “חתימה” HMAC אופציונלית)
# -*- coding: utf-8 -*-
import os, hashlib, json, hmac
from typing import Optional

CAS_ROOT = os.environ.get("IMU_CAS", "var/cas")

def _p(hash_hex: str) -> str:
    return os.path.join(CAS_ROOT, hash_hex[:2], hash_hex[2:4], hash_hex)

def put_bytes(b: bytes) -> str:
    h = hashlib.sha256(b).hexdigest()
    p = _p(h)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if not os.path.exists(p):
        with open(p, "wb") as f:
            f.write(b)
    return h

def put_json(obj) -> str:
    b = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return put_bytes(b)

def get(hash_hex: str) -> Optional[bytes]:
    p = _p(hash_hex)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return f.read()

def sign_hmac(hash_hex: str, key: bytes) -> str:
    """
    חתימה סימטרית פשוטה על ה-hash (HMAC-SHA256). אין תלות חיצונית.
    """
    return hmac.new(key, hash_hex.encode("ascii"), digestmod="sha256").hexdigest()

def verify_hmac(hash_hex: str, sig_hex: str, key: bytes) -> bool:
    return hmac.compare_digest(sign_hmac(hash_hex, key), sig_hex)