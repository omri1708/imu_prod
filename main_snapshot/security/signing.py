# imu_repo/security/signing.py
from __future__ import annotations
import os, json, hmac, hashlib
from typing import Dict, Any
from security.ed25519_optional import ed25519_available, ed25519_sign, ed25519_verify

KEYS_PATH = os.environ.get("IMU_KEYS_PATH", "/mnt/data/.imu_keys.json")
# מבנה קובץ המפתחות (JSON):
# {
#   "default": {"alg":"HMAC", "secret":"hex..."},
#   "prodKey": {"alg":"Ed25519", "pub":"hex...", "priv":"hex..."}
# }

class SignError(Exception): ...

def _load_keys() -> Dict[str,Any]:
    if not os.path.exists(KEYS_PATH):
        # צור מפתח HMAC ברירת־מחדל אם אין קובץ.
        secret = os.urandom(32).hex()
        doc = {"default": {"alg":"HMAC", "secret": secret}}
        with open(KEYS_PATH, "w") as f: json.dump(doc, f, indent=2)
        return doc
    with open(KEYS_PATH, "r") as f: return json.load(f)

def _save_keys(doc: Dict[str,Any]) -> None:
    tmp = KEYS_PATH + ".tmp"
    with open(tmp, "w") as f: json.dump(doc, f, indent=2)
    os.replace(tmp, KEYS_PATH)

def ensure_ed25519_key(key_id: str) -> None:
    if not ed25519_available():
        raise SignError("pynacl not available for Ed25519")
    doc = _load_keys()
    if key_id in doc and doc[key_id].get("alg") == "Ed25519":
        return
    # צור מפתח חדש
    from security.ed25519_optional import ed25519_keygen
    pub, priv = ed25519_keygen()
    doc[key_id] = {"alg":"Ed25519", "pub":pub, "priv":priv}
    _save_keys(doc)

def sign_manifest(payload: Dict[str,Any], *, key_id: str="default") -> Dict[str,Any]:
    doc = _load_keys()
    key = doc.get(key_id) or doc["default"]
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    if key.get("alg") == "Ed25519":
        sig = ed25519_sign(key["priv"], data)
        return {"payload": payload, "signature": {"alg":"Ed25519","key_id":key_id,"sig":sig}}
    # HMAC fallback
    secret = bytes.fromhex(key.get("secret") or doc["default"]["secret"])
    mac = hmac.new(secret, data, hashlib.sha256).hexdigest()
    return {"payload": payload, "signature": {"alg":"HMAC-SHA256","key_id":key_id,"mac":mac}}

def verify_manifest(signed: Dict[str,Any]) -> None:
    sig = signed.get("signature") or {}
    payload = signed.get("payload")
    if payload is None: raise SignError("missing payload")
    alg = sig.get("alg","")
    key_id = sig.get("key_id","default")
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    doc = _load_keys()
    key = doc.get(key_id) or doc.get("default")
    if not key: raise SignError("key missing")
    if alg == "Ed25519":
        pub = key.get("pub")
        if not pub: raise SignError("missing pub for Ed25519")
        ok = ed25519_verify(pub, data, sig.get("sig",""))
        if not ok: raise SignError("bad ed25519 signature")
        return
    # HMAC
    secret = bytes.fromhex(key.get("secret") or doc["default"]["secret"])
    expected = hmac.new(secret, data, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig.get("mac","")):
        raise SignError("bad HMAC")
    
class SignRequirementsError(SignError): ...
REQUIRE_ED25519 = os.environ.get("IMU_REQUIRE_ED25519","0") == "1"

def _require_ed25519_if_prod():
    if REQUIRE_ED25519 and not ed25519_available():
        raise SignRequirementsError("Ed25519 required (IMU_REQUIRE_ED25519=1) but pynacl not available")

def sign_manifest(payload: Dict[str,Any], *, key_id: str="default") -> Dict[str,Any]:
    _require_ed25519_if_prod()