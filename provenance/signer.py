# provenance/signer.py
import hmac, hashlib, json, os, time
from typing import Dict, Any

# מפתח סימטרי ב־env (ללא תלות חיצונית). אפשר להחליף ב־Ed25519 אם תרצה, בעזרת ספרייה ייעודית.
SECRET_ENV = "IMU_HMAC_KEY"

def _secret() -> bytes:
    key = os.environ.get(SECRET_ENV)
    if not key:
        # מפתח דיפולטי־למכונה; בפרודקשן חובה לקבוע env
        key = "change-me-in-production"
    return key.encode("utf-8")

def sign_record(record: Dict[str, Any]) -> Dict[str, Any]:
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    sig = hmac.new(_secret(), payload, hashlib.sha256).hexdigest()
    rec = dict(record)
    rec["signature"] = sig
    rec["key_id"] = "hmac-sha256"
    return rec

def verify_hmac(record_with_sig: Dict[str, Any]) -> bool:
    rec = dict(record_with_sig)
    sig = rec.pop("signature", None)
    if not sig:
        return False
    payload = json.dumps(rec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    expected = hmac.new(_secret(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, expected)

def signed_evidence(digest: str, source: str, trust: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    base = {
        "digest": digest,
        "source": source,
        "trust_hint": trust,
        "metadata": metadata,
        "timestamp_s": int(time.time())
    }
    return sign_record(base)