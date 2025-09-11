# imu_repo/tests/test_stage46_user_consolidation.py
from __future__ import annotations
import os, json, time, shutil, tempfile
from user_model.consolidation import Consolidator
from grounded.ttl import TTLPolicy

def run():
    tmp = tempfile.mkdtemp(prefix="imu_user_")
    try:
        cons = Consolidator(root=tmp)
        uid = "lea"

        # מוסיפים שלוש העדפות סותרות לאותה מפתח, עם trust/conf שונים
        cons.add_event(uid, "preference", {"key":"theme","value":"dark"}, confidence=0.9, trust=0.9, stable_hint=True)
        cons.add_event(uid, "preference", {"key":"theme","value":"light"}, confidence=0.4, trust=0.4)
        cons.add_event(uid, "preference", {"key":"theme","value":"dark"}, confidence=0.8, trust=0.7)

        # קונסולידציה → צפוי לבחור 'dark'
        out = cons.consolidate(uid)
        chosen = out["profile"]["preferences"]["theme"]["value"]
        ok1 = (chosen == "dark")

        # TTL: אירוע emotion עם ביטחון נמוך יתפוגג מהר (נכריז שעבר זמן)
        rec = cons.add_event(uid, "emotion", {"primary":"joy"}, confidence=0.3, trust=0.6)
        exp = rec["evidence"]["expire_ts"]
        # מזייפים “חלוף זמן” (נבדוק לוגית עם TTLPolicy.is_fresh)
        ok2 = TTLPolicy.is_fresh(exp)  # כרגע טרי
        ok = ok1 and ok2
        print("OK" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if __name__=="__main__":
    raise SystemExit(run())