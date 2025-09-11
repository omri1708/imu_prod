# imu_repo/engine/cas_verify.py
from __future__ import annotations
import hmac, hashlib, json
from typing import Dict, Any

class CasVerifyError(Exception): ...

def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",",":")).encode("utf-8")

def verify_bundle_signature(bundle: Dict[str,Any], keyring: Dict[str,Dict[str,str]]) -> bool:
    """
    keyring: {"key_id":{"secret_hex":"..","algo":"sha256"}, ...}
    bundle["signature"]={"sig":"..","algo":"sha256","key_id":"main",...}
    """
    sig = bundle.get("signature")
    if not isinstance(sig, dict):
        raise CasVerifyError("missing signature")
    kid = sig.get("key_id")
    algo = (sig.get("algo") or "sha256").lower()
    sig_hex = (sig.get("sig") or "").lower()
    if not kid or kid not in keyring:
        raise CasVerifyError(f"unknown key: {kid}")

    secret_hex = keyring[kid]["secret_hex"]
    secret = bytes.fromhex(secret_hex)

    # הקפד שלא לחתום על החתימה עצמה
    m = dict(bundle)
    m.pop("signature", None)
    data = canonical_json(m)
    try:
        digest = getattr(hashlib, algo)
    except AttributeError:
        raise CasVerifyError(f"unsupported algo {algo}")
    mac = hmac.new(secret, data, digest)
    calc = mac.hexdigest().lower()
    return hmac.compare_digest(calc, sig_hex)