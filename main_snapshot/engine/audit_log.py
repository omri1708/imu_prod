# imu_repo/engine/audit_log.py
from __future__ import annotations
import os, json, hashlib, hmac, time, uuid
from typing import Any, Dict, Optional, List
from security.signing import sign_manifest, verify_manifest

from engine.cas_store import put_json

AUDIT_PATH = os.environ.get("IMU_AUDIT_LOG", "/mnt/data/imu_audit.log.jsonl")

class AuditError(Exception): ...



def _audit_root() -> str:
    return os.environ.get("IMU_AUDIT_DIR", os.path.abspath(".imu_audit"))

def _log_path() -> str:
    return os.path.join(_audit_root(), "audit.log.jsonl")

def _ensure():
    os.makedirs(_audit_root(), exist_ok=True)

def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def _sha256_hex(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def _sign_entry(entry_body: Dict[str,Any]) -> Dict[str,Any]:
    """
    חותם את גוף האירוע. אם IMU_AUDIT_KEY קיים — HMAC-SHA256.
    אחרת — fingerprint = SHA256(body).
    """
    key = os.environ.get("IMU_AUDIT_KEY")
    canon = _canonical(entry_body)
    if key:
        sig = hmac.new(key.encode("utf-8"), canon, hashlib.sha256).hexdigest()
        return {"mode": "hmac-sha256", "value": sig}
    else:
        return {"mode": "sha256", "value": _sha256_hex(canon)}

def record_event(
    event: str,
    payload: Dict[str,Any],
    *,
    severity: str = "info",
    snap_before: Optional[Dict[str,Any]] = None,
    snap_after: Optional[Dict[str,Any]] = None,
    changed_paths: Optional[List[str]] = None
) -> Dict[str,Any]:
    """
    רושם אירוע ל־audit.log.jsonl עם חתימה ו־CAS Snapshots (אם ניתנו).
    מחזיר את גוף הרשומה (כולל cas refs).
    """
    _ensure()
    ts = time.time()
    entry_id = str(uuid.uuid4())
    cas_refs: Dict[str,Any] = {}
    if snap_before is not None:
        cas_refs["before"] = put_json(snap_before)
    if snap_after is not None:
        cas_refs["after"] = put_json(snap_after)
    body = {
        "id": entry_id,
        "ts": ts,
        "event": event,
        "severity": severity,
        "payload": payload or {},
        "cas": cas_refs or None,
        "changed_paths": changed_paths or None,
        "version": 1
    }
    sig = _sign_entry(body)
    entry = {"signature": sig, **body}
    with open(_log_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry





def _now_ts() -> int: return int(time.time())

def _read_last_record() -> Optional[Dict[str,Any]]:
    if not os.path.exists(AUDIT_PATH): return None
    last = None
    with open(AUDIT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last = json.loads(line)
    return last

def _record_event(action: str, details: Dict[str,Any], *, severity: str="info") -> Dict[str,Any]:
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