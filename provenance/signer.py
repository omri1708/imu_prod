# provenance/signer.py
# -*- coding: utf-8 -*-
"""
Provenance קשיח: CAS (sha256) + חתימה סימטרית (HMAC-SHA256) + רמות אמון.
ללא תלות חיצונית. ניתן להחליף ל-Ed25519 בהמשך אם תרצה ספרייה קריפטוגרפית.
"""

from __future__ import annotations
import os, json, time, hmac, hashlib, base64
from typing import Dict, Any

CAS_DIR = os.path.abspath("./.imu/cas")
SIG_DIR = os.path.abspath("./.imu/signatures")
KEY_DIR = os.path.abspath("./.imu/keys")
os.makedirs(CAS_DIR, exist_ok=True)
os.makedirs(SIG_DIR, exist_ok=True)
os.makedirs(KEY_DIR, exist_ok=True)

def _path_for_digest(d: str) -> str:
    return os.path.join(CAS_DIR, d)

def _b(s: str) -> bytes:
    return s.encode("utf-8")

def put_blob(data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()
    p = _path_for_digest(digest)
    if not os.path.exists(p):
        with open(p, "wb") as f:
            f.write(data)
    return digest

def get_blob(digest: str) -> bytes:
    p = _path_for_digest(digest)
    with open(p, "rb") as f:
        return f.read()

def save_json(obj: Dict[str, Any]) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return put_blob(data)

def load_json(digest: str) -> Dict[str, Any]:
    data = get_blob(digest)
    return json.loads(data.decode("utf-8"))

def _key_path(user_id: str) -> str:
    return os.path.join(KEY_DIR, f"{user_id}.key")

def ensure_key(user_id: str) -> bytes:
    kp = _key_path(user_id)
    if os.path.exists(kp):
        return open(kp, "rb").read()
    key = hashlib.sha256(_b(f"seed::{user_id}::{time.time()}")).digest()
    with open(kp, "wb") as f:
        f.write(key)
    return key

def sign(user_id: str, digest: str, trust_score: float = 0.5) -> str:
    """
    חותם ערך CAS עם מפתח HMAC פר-משתמש + trust_score. מחזיר מזהה חתימה.
    """
    key = ensure_key(user_id)
    body = json.dumps({"digest": digest, "user": user_id, "ts": time.time(), "trust": trust_score},
                      sort_keys=True, separators=(",", ":")).encode("utf-8")
    mac = hmac.new(key, body, hashlib.sha256).digest()
    sig = base64.urlsafe_b64encode(mac + body).decode("utf-8")
    # נשמור גם בקובץ ל-Audit
    sig_id = hashlib.sha256(_b(sig)).hexdigest()
    with open(os.path.join(SIG_DIR, sig_id + ".sig"), "w", encoding="utf-8") as f:
        f.write(sig)
    return sig_id

def verify_signature(user_id: str, sig_id: str) -> bool:
    path = os.path.join(SIG_DIR, sig_id + ".sig")
    if not os.path.exists(path):
        return False
    sig = open(path, "r", encoding="utf-8").read()
    raw = base64.urlsafe_b64decode(sig.encode("utf-8"))
    mac, body = raw[:32], raw[32:]
    key = ensure_key(user_id)
    calc = hmac.new(key, body, hashlib.sha256).digest()
    return hmac.compare_digest(mac, calc)

def record_evidence(user_id: str, payload: Dict[str, Any], trust_hint: float = 0.5) -> Dict[str, Any]:
    """
    יוצר רשומת ראיה: מכניס ל-CAS, חותם, מחזיר מטא-דאטה מלא.
    """
    digest = save_json(payload)
    sig_id = sign(user_id, digest, trust_hint)
    return {"digest": digest, "sig_id": sig_id, "user": user_id, "ts": time.time(), "trust": trust_hint}

#TODO מפתח סימטרי ב־env (ללא תלות חיצונית). אפשר להחליף ב־Ed25519 אם תרצה, בעזרת ספרייה ייעודית.
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