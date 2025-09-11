# imu_repo/tests/test_stage53_user_sync.py
from __future__ import annotations
import time
from user_model.identity import ensure_user
from user_model.memory_store import put_event, get_profile, forget
from user_model.sync_protocol import export_snapshot, import_and_merge

def run():
    u = ensure_user("sync@example.com"); uid = u["uid"]
    # בנה ידע מקומי
    put_event(uid, "pref", "dark_mode", True, confidence=0.7, source="ui")
    put_event(uid, "belief", "quality_strict", 0.8, confidence=0.9, source="policy")
    p1 = get_profile(uid)

    # יצוא מוצפן → "מכשיר" שני: ננקה ואז נייבא
    snap = export_snapshot(uid)
    forget(uid)  # מנקה T0/T1/T2
    import_and_merge(uid, snap)
    p2 = get_profile(uid)

    ok = (abs(p1["pref"]["dark_mode"] - p2["pref"]["dark_mode"]) < 1e-6) and \
         (abs(p1["beliefs"]["quality_strict"] - p2["beliefs"]["quality_strict"]) < 1e-6)
    print("OK" if ok else "FAIL")
    return 0 if ok else 1

if __name__=="__main__":
    raise SystemExit(run())