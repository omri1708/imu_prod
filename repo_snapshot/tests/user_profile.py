# imu_repo/tests/user_profile.py
from __future__ import annotations
import os, json, time
from user.auth import UserStore
from user.consciousness import UserMind
from user.memory_state import MemoryState

def run():
    us=UserStore()
    us.ensure_user("alice", roles=["user"], consent={"memory":True,"analytics":True})
    us.ensure_user("bob", roles=["admin"], consent={"memory":True,"analytics":False})
    tok=us.issue_token("alice", ttl_s=60)
    assert us.verify_token(tok)=="alice"

    mA=MemoryState("alice"); mindA=UserMind("alice", mA)
    mindA.observe_emotion("frustrated", 0.9)
    mindA.set_goal("build_any_app", True, 0.9)
    mindA.assert_belief("pref_ui_theme","dark",0.8)
    mindA.assert_belief("pref_ui_theme","light",0.6)  # לא יחליף כי conf נמוך/לא טרי
    q = mindA.recall("ui theme")
    c = mindA.consolidate()
    mindA.decay()
    hints = mindA.routing_hints()
    print("alice hints:", hints)
    print("alice recall:", [r["text"] for r in q][:3], "consolidated:", c)

    # מחיקה לפי פרטיות (זכות להישכח)
    mA.erase({"kind":"belief","meta":{"key":"pref_ui_theme"}})
    q2 = mindA.recall("ui theme")
    print("after erase:", [r["text"] for r in q2][:3])

    # bob
    mB=MemoryState("bob"); mindB=UserMind("bob", mB)
    mindB.observe_emotion("calm", 0.7)
    hintsB=mindB.routing_hints()
    print("bob hints:", hintsB)

    return 0

if __name__=="__main__":
    raise SystemExit(run())