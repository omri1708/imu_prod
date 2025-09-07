# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple, Dict, Any

def normalize_build_request(free_text:str, slots:Dict[str,Any]) -> Tuple[str,Dict[str,Any]]:
    """
    מחלץ כוונה לבנות: סוג (web/cli/mobile/desktop), שם, פלטפורמה. אם חסר — נחזיר שאלה ידידותית.
    """
    s=slots.copy()
    # סוג ברירת מחדל לפי טקסט – מפשט: "אתר"→web, "אפליקציה"→web, "כלי"→cli
    low=free_text.lower()
    if "אתר" in free_text or "site" in low or "web" in low: s.setdefault("type","web")
    if "כלי" in free_text or "cli" in low: s.setdefault("type","cli")
    if "mobile" in low or "android" in low or "ios" in low: s.setdefault("type","mobile")
    if "desktop" in low or "unity" in low: s.setdefault("type","desktop")

    s.setdefault("name","myapp")

    # החזר שאלה אם חסר
    if "type" not in s:
        return ("איזה סוג תרצה? (אתר / כלי שורת־פקודה / מובייל / דסקטופ)", s)
    t=s["type"]
    if t=="web":
        s.setdefault("stack","python_web")  # Flask מינימלי; אפשר להחליף ל-node/go בהמשך
    elif t=="cli":
        s.setdefault("stack","python_app")
    elif t in ("mobile","desktop"):
        # נבקש הרכבות עתידיות; כרגע נחזיר בקשת משאבים אמיתית
        return ("מוכן. לפיתוח {} נזדקק ל-SDKs (Android/iOS/Unity). תרצה שאתן הוראות התקנה או שנתחיל מגרסה Web/CLI זמינה?".format("מובייל" if t=="mobile" else "דסקטופ"), s)
    return ("מצוין. אבנה '{}' מסוג {} (stack: {}). לאשר?".format(s["name"], s["type"], s["stack"]), s)
