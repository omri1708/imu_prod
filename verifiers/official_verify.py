# imu_repo/verifiers/official_verify.py
from __future__ import annotations
import json
from typing import Dict, Any, Tuple
from verifiers.official_registry import get_official, hmac_sha256

def verify_official_payload(payload: Dict[str, Any]) -> Tuple[bool, str]:
    """
    payload צפוי להכיל:
      - "data": אובייקט מסומן
      - "official": { "source_id": str, "signature": str }
    אימות HMAC-SHA256 מול הסוד הרשום של המקור.
    """
    off = payload.get("official", {})
    src = str(off.get("source_id", ""))
    sig = str(off.get("signature", ""))
    if not src or not sig:
        return False, "missing_signature"

    rec = get_official(src)
    if rec is None:
        return False, "unknown_official_source"

    data = payload.get("data")
    blob = json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
    expect = hmac_sha256(rec["shared_secret"], blob)
    if expect != sig:
        return False, "bad_signature"
    return True, "ok"