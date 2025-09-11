# imu_repo/user_model/sync_protocol.py
from __future__ import annotations
from typing import Dict, Any, List
import os, json, time, hmac, hashlib
from user_model.identity import user_dir, load_key
from user_model.crypto_store import encrypt_bytes, decrypt_bytes
from user_model.memory_store import list_events, rebuild_from_events
from user_model.event_crdt import gset_union

SYNC_ROOT = "/mnt/data/imu_repo/user_sync"

def _nonce(uid: str) -> bytes:
    return f"IMU_SYNC_NONCE__{uid}".encode("utf-8")

def _hmac(key: bytes, payload: bytes) -> str:
    return hmac.new(key, payload, hashlib.sha256).hexdigest()

def export_snapshot(uid: str) -> str:
    """
    יוצר צילום מוצפן/חתום של אירועי המשתמש (T0 בלבד). מוצפן במפתח המשתמש.
    """
    os.makedirs(SYNC_ROOT, exist_ok=True)
    events = list_events(uid)
    blob = json.dumps({"uid": uid, "ts": time.time(), "events": events},
                      ensure_ascii=False, sort_keys=True, separators=(",",":")).encode("utf-8")
    key = load_key(uid)
    ct  = encrypt_bytes(key, blob, nonce=_nonce(uid))
    sig = _hmac(key, ct)
    path = os.path.join(SYNC_ROOT, f"{uid}_{int(time.time())}.imu.enc")
    with open(path, "wb") as f: f.write(ct + b"." + sig.encode("utf-8"))
    return path

def import_and_merge(uid: str, snapshot_path: str) -> Dict[str,Any]:
    """
    קורא צילום מוצפן, מאמת חתימה, מאחד G-Set של אירועים, ובונה מחדש T1/T2.
    """
    key = load_key(uid)
    with open(snapshot_path, "rb") as f:
        raw = f.read()
    try:
        ct, sig = raw.rsplit(b".", 1)
    except ValueError:
        raise RuntimeError("snapshot_format_invalid")
    if _hmac(key, ct).encode("utf-8") != sig:
        raise RuntimeError("snapshot_hmac_invalid")

    pt = decrypt_bytes(key, ct, nonce=_nonce(uid))
    obj = json.loads(pt.decode("utf-8"))
    if obj.get("uid") != uid:
        raise RuntimeError("snapshot_uid_mismatch")

    # איחוד G-Set עם אירועים מקומיים
    local = list_events(uid)
    merged = gset_union(local, obj.get("events", []))
    rebuild_from_events(uid, merged)
    return {"merged_count": len(merged), "local_count": len(local)}