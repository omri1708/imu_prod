# imu_repo/security/response_signer.py
from __future__ import annotations
import os, hmac, hashlib, json, time, base64
from typing import Any, Dict

KEYF = "/mnt/data/imu_repo/keys/resp_sign.key"
os.makedirs(os.path.dirname(KEYF), exist_ok=True)

def _key() -> bytes:
    if not os.path.exists(KEYF):
        k = os.urandom(32)
        open(KEYF, "wb").write(k)
        return k
    return open(KEYF, "rb").read()

def sign_payload(payload_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    מקבל אובייקט תשובה (למשל {"text":..., "claims":[...]})
    ומחזיר אותו עם שדות חתימה: {"sig":{"alg":"HMAC-SHA256","ts":..., "mac":"..."}}
    החתימה נעשית על בסיס JSON קנוני (separators, sort_keys).
    """
    ts = int(time.time()*1000)
    canonical = json.dumps(payload_obj, ensure_ascii=False, sort_keys=True, separators=(",",":")).encode("utf-8")
    mac = hmac.new(_key(), canonical + str(ts).encode("ascii"), hashlib.sha256).digest()
    out = dict(payload_obj)
    out["sig"] = {"alg":"HMAC-SHA256","ts":ts,"mac": base64.b64encode(mac).decode("ascii")}
    return out

def verify_payload(payload_obj: Dict[str, Any]) -> bool:
    """
    אימות בצד לקוח/טסט: בודק mac מול התוכן (ללא שדה sig) + ts.
    """
    sig = payload_obj.get("sig") or {}
    ts = sig.get("ts"); mac_b64 = sig.get("mac")
    if ts is None or mac_b64 is None: return False
    po = dict(payload_obj); po.pop("sig", None)
    canonical = json.dumps(po, ensure_ascii=False, sort_keys=True, separators=(",",":")).encode("utf-8")
    try:
        mac = base64.b64decode(mac_b64.encode("ascii"))
    except Exception:
        return False
    calc = hmac.new(_key(), canonical + str(ts).encode("ascii"), hashlib.sha256).digest()
    # הגנה מול השוואה זולה
    return hmac.compare_digest(mac, calc)