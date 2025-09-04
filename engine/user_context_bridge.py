# imu_repo/engine/user_context_bridge.py
from __future__ import annotations
from typing import Dict, Any, List
from user_model.semantic_store import search
from user_model.conflict_resolution import resolve_preferences

def load_user_context(user_key: str) -> Dict[str,Any]:
    """
    מחלץ העדפות 'top-of-mind' ע"י חיפוש סמנטי קצר על 'preferences'.
    """
    prefs = search(user_key, "preferences settings theme language layout", topk=10, purpose="preferences")
    # מקבץ per-key — בדוגמה נניח שהטקסט כולל תבנית: "pref:key=value"
    by_key: Dict[str, List[Dict[str,Any]]] = {}
    for s, rec in prefs:
        # אנו שומרים meta בלבד. קרא טקסט נדרש? אפשר להרחיב — כאן נשאר על meta (vec)
        # לצורך דמו החלטה, נניח meta מכילה pseudo 'kv' אם נרשם כך.
        kv = rec.get("kv")  # אופציונלי; אם לא קיים, מתעלמים
        if not kv: continue
        by_key.setdefault(kv["key"], []).append(rec)
    decided={}
    for k, cands in by_key.items():
        r = resolve_preferences(cands)
        if r["decided"]:
            decided[k] = r["winner"]["kv"]["value"]
    return {"preferences": decided}

def update_user_context(user_key: str, decisions: Dict[str,Any]) -> None:
    # hook להזרמת החלטות חזרה לזיכרון/טלאים; כאן נשאיר כ-noop (כדי לא לכפות כתיבה בפועל)
    return None

#TODO
# הערה: אם תרצה שמנוע הסינתזה יכתוב רשומות kv אמיתיות 
# (למשל "kv":{"key":"theme","value":"dark"}) —
#  הוסף זאת בשלב 37/34 בזמן שנשמרת תצפית על העדפה.