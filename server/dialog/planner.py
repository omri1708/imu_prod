# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple, Dict, Any
import os, json

LLM_ENABLED = os.getenv("LLM_ENABLED","0") == "1"
_llm = None
if LLM_ENABLED:
    try:
        from integration.llm_client import LLMClient
        _llm = LLMClient()
    except Exception:
        LLM_ENABLED = False

def normalize_build_request(free_text: str, slots: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    מחלץ בקשה לבנייה: type (web/cli/mobile/desktop), name, stack (python_web/python_app).
    אם חסר מידע – מחזיר שאלה. אם LLM פעיל, ממלא מראש JSON לפי subject_profile.
    """
    s = dict(slots or {})

    # 1) LLM (אם פעיל) – JSON קשיח בלבד
    if LLM_ENABLED and _llm:
        prof = s.get("subject_profile") or {}
        sys = ("החזר JSON תקין בלבד: {name, type in [web,cli,mobile,desktop], stack in [python_web,python_app]}. "
               "התאם לבקשת המשתמש ולהעדפות בפרופיל. אין טקסט חופשי.")
        try:
            draft = _llm.chat(
                [{"role":"system","content":sys},
                 {"role":"user","content":"פרופיל:\n"+json.dumps(prof,ensure_ascii=False)+"\n\nבקשה:\n"+(free_text or "")}],
                temperature=0.2, max_tokens=200)
            data = json.loads(draft)
            if isinstance(data, dict):
                if data.get("name"):  s["name"]=data["name"]
                if data.get("type") in ("web","cli","mobile","desktop"): s["type"]=data["type"]
                if data.get("stack") in ("python_web","python_app"):     s["stack"]=data["stack"]
        except Exception:
            pass

    # 2) היוריסטיקה משלימה
    low = (free_text or "").lower()
    if "אתר" in (free_text or "") or "web" in low or "site" in low: s.setdefault("type","web")
    if "cli" in low or "כלי" in (free_text or ""): s.setdefault("type","cli")
    if any(k in low for k in ("mobile","android","ios")): s.setdefault("type","mobile")
    if any(k in low for k in ("desktop","unity")): s.setdefault("type","desktop")
    s.setdefault("name","app")

    # 3) אם עדיין חסר – שאלה
    if "type" not in s:
        return ("איזה סוג אפליקציה לבנות? (web / cli / mobile / desktop)", s)

    # 4) stack ברירת מחדל
    if s["type"] == "web": s.setdefault("stack","python_web")
    elif s["type"] == "cli": s.setdefault("stack","python_app")
    elif s["type"] in ("mobile","desktop"):
        return (f"מוכן. לפיתוח {s['type']} צריך SDKs (Android/iOS/Unity). כרגע זמינה רק WEB/CLI. להמשיך ב-web/CLI זמינה?", s)

    # 5) אישור סופי
    return (f"מצוין. אבנה '{s['name']}' מסוג {s['type']} (stack: {s['stack']}). לאשר?", s)
