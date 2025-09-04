# imu_repo/engine/audit_log.py
from __future__ import annotations
import os, json, time, hashlib
from typing import Dict, Any, Optional
from security.signing import sign_manifest, verify_manifest

AUDIT_PATH = os.environ.get("IMU_AUDIT_LOG", "/mnt/data/imu_audit.log.jsonl")

class AuditError(Exception): ...

def _now_ts() -> int: return int(time.time())

def _read_last_record() -> Optional[Dict[str,Any]]:
    if not os.path.exists(AUDIT_PATH): return None
    last = None
    with open(AUDIT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last = json.loads(line)
    return last

def record_event(action: str, details: Dict[str,Any], *, severity: str="info") -> Dict[str,Any]:
    """
    רושם אירוע חתום, כולל hash של הרשומה הקודמת לצורך שרשור.
    """
    prev = _read_last_record()
    prev_hash = ""
    if prev:
        blob = json.dumps(prev, sort_keys=True).encode("utf-8")
        prev_hash = hashlib.sha256(blob).hexdigest()

    payload = {
        "ts": _now_ts(),
        "severity": severity,
        "action": action,
        "details": details,
        "prev_hash": prev_hash,
        "v": 1
    }
    signed = sign_manifest(payload, key_id=os.environ.get("IMU_AUDIT_KEY","default"))
    os.makedirs(os.path.dirname(AUDIT_PATH), exist_ok=True)
    with open(AUDIT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(signed, ensure_ascii=False) + "\n")
    return signed

def verify_chain() -> Dict[str,Any]:
    """
    מאמת את שרשרת החתימות והקישורים בין רשומות.
    """
    if not os.path.exists(AUDIT_PATH):
        return {"ok": True, "count": 0}
    prev_signed = None
    prev_hash = ""
    count = 0
    with open(AUDIT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            signed = json.loads(line)
            verify_manifest(signed)  # אימות חתימה של הרשומה עצמה
            blob = json.dumps(signed, sort_keys=True).encode("utf-8")
            curr_hash = hashlib.sha256(blob).hexdigest()
            payload = signed["payload"]
            if prev_signed:
                if payload.get("prev_hash","") != prev_hash:
                    return {"ok": False, "error": "chain break", "at": count}
            prev_hash = curr_hash
            prev_signed = signed
            count += 1
    return {"ok": True, "count": count}