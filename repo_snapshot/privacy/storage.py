# imu_repo/privacy/storage.py
from __future__ import annotations
import os, json, time
from typing import Any, Optional
from privacy.keystore import encrypt_bytes, decrypt_bytes, BASE

def _u_dir(user_id: str) -> str:
    p = os.path.join(BASE, user_id)
    os.makedirs(p, exist_ok=True)
    return p

def save_json_encrypted(user_id: str, name: str, obj: Any, *, ttl_s: Optional[float] = None) -> str:
    p = os.path.join(_u_dir(user_id), f"{name}.sealed.json")
    meta = {"ts": time.time(), "ttl_s": float(ttl_s) if ttl_s is not None else None}
    blob = json.dumps({"meta": meta, "data": obj}, ensure_ascii=False).encode("utf-8")
    sealed = encrypt_bytes(user_id, blob)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(sealed, f, ensure_ascii=False)
    return p

def load_json_encrypted(user_id: str, name: str) -> Optional[Any]:
    p = os.path.join(_u_dir(user_id), f"{name}.sealed.json")
    if not os.path.exists(p):
        return None
    try:
        sealed = json.load(open(p, "r", encoding="utf-8"))
        blob = decrypt_bytes(user_id, sealed)
        pkt = json.loads(blob.decode("utf-8"))
        meta, data = pkt.get("meta", {}), pkt.get("data")
        ts = float(meta.get("ts", 0))
        ttl = meta.get("ttl_s")
        if ttl is not None and time.time() > ts + float(ttl):
            # פג — נמחק
            try: os.remove(p)
            except Exception: pass
            return None
        return data
    except Exception:
        return None

def purge_expired(user_id: str) -> int:
    """מוחק קבצים שפגו. מחזיר כמה נמחקו."""
    cnt = 0
    d = _u_dir(user_id)
    now = time.time()
    for fn in os.listdir(d):
        if not fn.endswith(".sealed.json"):
            continue
        p = os.path.join(d, fn)
        try:
            sealed = json.load(open(p, "r", encoding="utf-8"))
            blob = decrypt_bytes(user_id, sealed)
            pkt = json.loads(blob.decode("utf-8"))
            meta = pkt.get("meta", {})
            ts = float(meta.get("ts", 0))
            ttl = meta.get("ttl_s")
            if ttl is not None and now > ts + float(ttl):
                os.remove(p)
                cnt += 1
        except Exception:
            # לא מצליחים לפרש — נשאיר (fail-open לטובת חקירה)
            pass
    return cnt