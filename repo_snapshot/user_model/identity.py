# imu_repo/user_model/identity.py
from __future__ import annotations
from typing import Dict, Any
import os, json, hashlib, hmac, time, secrets, shutil
import base64

ROOT = "/mnt/data/imu_repo/users"
SEED = os.environ.get("IMU_USER_SEED", "imu_user_seed_dev")


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _uid_from_str(s: str) -> str:
    """user_id דטרמיניסטי נטול רגישות (sha256-12)."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

def user_dir(user_key: str) -> str:
    _ensure_dir(ROOT)
    uid = _uid_from_str(user_key)
    up = os.path.join(ROOT, uid)
    _ensure_dir(up)
    return up

def ensure_master_key(user_key: str) -> bytes:
    """מייצר/טוען מפתח ראשי פר-משתמש (לא משותף)."""
    up = user_dir(user_key)
    kp = os.path.join(up, "user.key")
    if not os.path.exists(kp):
        k = secrets.token_bytes(32)
        with open(kp, "wb") as f: f.write(k)
        open(os.path.join(up,"audit.log"),"a").write(json.dumps({"ts":time.time(),"op":"create_user"})+"\n")
        return k
    return open(kp,"rb").read()

def issue_token(user_key: str, *, ttl_s: int=3600) -> str:
    """טוקן חתום מקומית (HMAC-פשוט) — לשימוש פנימי בסדנבוקס."""
    up = user_dir(user_key)
    secret = ensure_master_key(user_key)
    now = int(time.time())
    payload = f"{_uid_from_str(user_key)}|{now}|{ttl_s}".encode()
    sig = hashlib.sha256(secret + payload).hexdigest()
    tok = base64.urlsafe_b64encode(payload + b"|" + sig.encode()).decode()
    return tok

def validate_token(user_key: str, token: str) -> bool:
    secret = ensure_master_key(user_key)
    try:
        raw = base64.urlsafe_b64decode(token.encode())
        uid, ts, ttl, sig = raw.decode().split("|")
        exp = hashlib.sha256(secret + f"{uid}|{ts}|{ttl}".encode()).hexdigest()
        if exp != sig: return False
        return (int(ts)+int(ttl)) >= int(time.time())
    except Exception:
        return False
def _ensure(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def stable_user_id(email_or_name: str) -> str:
    """זהות דטרמיניסטית (קשיח) ממשתנה קלט + seed."""
    msg = (email_or_name or "anon").encode("utf-8")
    key = SEED.encode("utf-8")
    uid = hmac.new(key, msg, hashlib.sha256).hexdigest()[:16]
    return uid

def keys_path(uid: str) -> str:
    return os.path.join(user_dir(uid), "keys.json")

def policy_path(uid: str) -> str:
    return os.path.join(user_dir(uid), "policy.json")

def meta_path(uid: str) -> str:
    return os.path.join(user_dir(uid), "meta.json")

def ensure_user(email_or_name: str, *, ttl_days: int=365, retain: bool=True) -> Dict[str,Any]:
    uid = stable_user_id(email_or_name)
    udir = user_dir(uid); _ensure(udir)
    # מפתח סימטרי פר-משתמש (לא תלוי חוץ)
    kpath = keys_path(uid)
    if not os.path.exists(kpath):
        key = secrets.token_hex(32)
        with open(kpath,"w",encoding="utf-8") as f:
            json.dump({"k": key}, f)
    # מדיניות ברירת מחדל
    if not os.path.exists(policy_path(uid)):
        with open(policy_path(uid),"w",encoding="utf-8") as f:
            json.dump({
                "quality": "standard",        # "strict" או "relaxed"
                "latency_p95_ms": 1500,       # יעד p95 ברירת מחדל
                "min_evidence_kinds": ["service_tests","perf_summary","ui_accessibility"],
                "min_evidence_count": 2
            }, f, ensure_ascii=False, indent=2)
    # פרטיות/TTL
    if not os.path.exists(meta_path(uid)):
        with open(meta_path(uid),"w",encoding="utf-8") as f:
            json.dump({
                "uid": uid,
                "created_at": time.time(),
                "ttl_days": int(ttl_days),
                "retain": bool(retain),
                "consent": {"store": True, "analytics": False}
            }, f, ensure_ascii=False, indent=2)
    return {"uid": uid, "dir": udir}

def load_key(uid: str) -> bytes:
    with open(keys_path(uid),"r",encoding="utf-8") as f:
        return bytes.fromhex(json.load(f)["k"])

def load_policy(uid: str) -> Dict[str,Any]:
    with open(policy_path(uid),"r",encoding="utf-8") as f:
        return json.load(f)

def save_policy(uid: str, policy: Dict[str,Any]) -> None:
    with open(policy_path(uid),"w",encoding="utf-8") as f:
        json.dump(policy, f, ensure_ascii=False, indent=2)

def forget_user(uid: str) -> None:
    """מחיקה קשיחה לפי בקשה."""
    p = user_dir(uid)
    if os.path.isdir(p):
        shutil.rmtree(p)