# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple, Dict, Any, List
import json, os
LLM_ENABLED = os.getenv("LLM_ENABLED","0")=="1"
if LLM_ENABLED:
    from integration.llm_client import LLMClient
    _llm = LLMClient()

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

def llm_build_spec(user_text: str, subject_profile: Dict[str,Any] | None = None) -> Dict[str,Any]:
    """
    מפיק BuildSpec ישיר (שם/שירותים/אילוצים) ל-orchestrator.build.
    שומר על אפס הלוצינציות: המודל מייצר *תכנית*, הביצוע בפועל נשאר דרך המנועים הדטרמיניסטיים.
    """
    if not LLM_ENABLED:
        # fallback: כמו היום – שירות יחיד בהתאם להיוריסטיקה
        _, s = normalize_build_request(user_text, {})
        return {"name": s.get("name","app"),
                "services":[{"type":"python_web" if s.get("stack")=="python_web" else "python_app",
                             "name": s.get("name","app")}]}
    prof = subject_profile or {}
    sys = ("את/ה מתכננ/ת תוכנה זהיר/ה. הפק JSON *תקין בלבד* לשדות:"
           " name, services[{type in [python_app,python_web], name}], constraints{...}."
           " אל תבטיח פעולה שלא ניתן לבצע. אין טקסט חופשי, JSON בלבד.")
    user = f"פרופיל:\n{json.dumps(prof,ensure_ascii=False)}\n\nבקשה:\n{user_text}\n"
    draft = _llm.chat([{"role":"system","content":sys},{"role":"user","content":user}], temperature=0.2, max_tokens=700)
    try:
        spec = json.loads(draft)
        # קשיחה: ודא פורמט מינימלי
        if not isinstance(spec.get('services'), list) or not spec.get('services'):
            spec['services'] = [{"type":"python_web","name": spec.get("name","app")}]
        return spec
    except Exception:
        # fallback בטוח
        return {"name":"app","services":[{"type":"python_web","name":"app"}]}