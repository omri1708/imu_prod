# imu_repo/security/signing.py
from __future__ import annotations
import os, hmac, hashlib, json
from typing import Dict, Any, Tuple


_KEYS_FILE = os.environ.get("IMU_KEYS_PATH", os.path.expanduser("~/.imu_keys.json"))

class KeyStoreError(Exception): ...
class VerifyError(Exception): ...

def _load_keys() -> Dict[str,str]:
    if not os.path.exists(_KEYS_FILE):
        return {}
    with open(_KEYS_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {}

def _save_keys(keys: Dict[str,str]) -> None:
    os.makedirs(os.path.dirname(_KEYS_FILE), exist_ok=True)
    tmp = _KEYS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(keys, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _KEYS_FILE)

def ensure_key(key_id: str="default") -> Tuple[str, bytes]:
    keys = _load_keys()
    if key_id not in keys:
        # 32 bytes hex secret
        secret = os.urandom(32).hex()
        keys[key_id] = secret
        _save_keys(keys)
    secret_hex = keys[key_id]
    return key_id, bytes.fromhex(secret_hex)

def _canon(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",",":")).encode("utf-8")

def sign_manifest(manifest: Dict[str, Any], *, key_id: str="default") -> Dict[str, Any]:
    """
    HMAC-SHA256 על ה-manifest הקנוני.
    מוסיף שדות: signature.alg, signature.key_id, signature.mac
    """
    _, key = ensure_key(key_id)
    body = {k: manifest[k] for k in manifest.keys()}  # shallow copy
    mac = hmac.new(key, _canon(body), hashlib.sha256).hexdigest()
    out = dict(body)
    out["signature"] = {"alg":"HMAC-SHA256","key_id": key_id, "mac": mac}
    return out

def verify_manifest(signed_manifest: Dict[str, Any]) -> None:
    """
    אם החתימה לא תואמת — זורק VerifyError
    """
    sig = signed_manifest.get("signature")
    if not sig or not isinstance(sig, dict):
        raise VerifyError("missing signature")
    key_id = sig.get("key_id")
    mac_got = sig.get("mac")
    keys = _load_keys()
    if key_id not in keys:
        raise VerifyError(f"unknown key_id: {key_id}")
    key = bytes.fromhex(keys[key_id])
    body = dict(signed_manifest)
    body.pop("signature", None)
    mac_exp = hmac.new(key, _canon(body), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(mac_got, mac_exp):
        raise VerifyError("bad signature")