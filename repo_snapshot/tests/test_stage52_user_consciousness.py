# imu_repo/tests/test_stage52_user_consciousness.py
from __future__ import annotations
import os, time
from user_model.identity import ensure_user, user_dir, forget_user
from user_model.memory_store import put_event, get_profile, garbage_collect
from user_model.crypto_store import load_encrypted_json
from user_model.identity import load_key
from user_model.memory_store import MEM_FILE, NONCE

def run():
    u = ensure_user("noa@example.com"); uid = u["uid"]
    # העדפות סותרות: dark_mode=True ואז False עם confidence/recency שונים
    put_event(uid, "pref", "dark_mode", True, confidence=0.6, ttl_s=3600, source="ui")
    time.sleep(0.02)
    put_event(uid, "pref", "dark_mode", False, confidence=0.9, ttl_s=3600, source="settings")
    prof = get_profile(uid)
    # צפוי משקל גבוה יותר ל-False → ממוצע מתחת 0.5
    val = prof["pref"]["dark_mode"]
    cond1 = (0.0 <= val <= 0.49)

    # TTL: ניצור אירוע קצר מועד
    put_event(uid, "pref", "banner_dismissed", True, confidence=0.9, ttl_s=0.01)
    time.sleep(0.02)
    garbage_collect(uid)
    prof2 = get_profile(uid)
    cond2 = ("banner_dismissed" not in prof2["pref"]) or (prof2["pref"]["banner_dismissed"] in (0.0, None))

    # הצפנה במנוחה: הקובץ לא נקרא כ-json ללא מפתח (קריאה מוצפנת ישירה תיכשל)
    p = os.path.join(user_dir(uid), MEM_FILE)
    ok_cipher = False
    try:
        open(p,"r",encoding="utf-8").read()  # אמור להחזיר אשפה/חריגה – אין הבטחה
        # נבדוק דה-קריפט:
        key = load_key(uid)
        obj = load_encrypted_json(p, key, nonce=NONCE)
        ok_cipher = isinstance(obj, dict) and "T0" in obj
    except Exception:
        # עדיין נבדוק דה-קריפט תקין:
        key = load_key(uid)
        obj = load_encrypted_json(p, key, nonce=NONCE)
        ok_cipher = isinstance(obj, dict) and "T0" in obj

    ok = cond1 and cond2 and ok_cipher
    print("OK" if ok else "FAIL")
    return 0 if ok else 1

if __name__=="__main__":
    raise SystemExit(run())