# imu_repo/engine/cas_sign.py
from __future__ import annotations
import hmac, hashlib, json, time
from typing import Dict, Any

class CassignError(Exception): ...

def _digest_for(algo: str):
    try:
        return getattr(hashlib, algo)
    except AttributeError:
        raise CassignError(f"unsupported hash algo: {algo}")

def canonical_json(obj: Any) -> bytes:
    # canonical form: UTF-8, separators, sort_keys=true
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",",":")).encode("utf-8")

def sign_manifest(manifest: Dict[str,Any], *, key_id: str, secret_hex: str, algo: str="sha256") -> Dict[str,Any]:
    """
    מחזיר בלוק חתימה שניתן להצמיד לכל חבילת CAS:
      { "sig": "...hex...", "algo":"sha256", "key_id":"kX", "signed_at": epoch }
    החתימה נעשית על canonical_json(manifest).
    """
    secret = bytes.fromhex(secret_hex)
    data = canonical_json(manifest)
    mac = hmac.new(secret, data, _digest_for(algo))
    return {
        "sig": mac.hexdigest(),
        "algo": algo,
        "key_id": key_id,
        "signed_at": time.time()
    }

def verify_manifest(manifest: Dict[str,Any], signature: Dict[str,Any], *, secret_hex: str) -> bool:
    algo = signature.get("algo","sha256")
    sig_hex = signature.get("sig") or ""
    secret = bytes.fromhex(secret_hex)
    data = canonical_json(manifest)
    mac = hmac.new(secret, data, _digest_for(algo))
    calc = mac.hexdigest()
    return hmac.compare_digest(calc.lower(), sig_hex.lower())