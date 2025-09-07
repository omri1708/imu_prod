# -*- coding: utf-8 -*-
from __future__ import annotations
import re, json
from typing import Tuple, Dict, Any

def _json_in(text:str):
    m=re.search(r"\{.*\}", text, re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None

def detect(message: str) -> Tuple[str, Dict[str,Any]]:
    """
    מחזיר (intent, slots) משפה חופשית (עברית/אנגלית) — בלי פקודות קשיחות.
    intents: greet, consent, build_app, ask_info, run_action, preference, unknown
    """
    m=(message or "").strip()
    low=m.lower()

    # ברכות/פתיחה
    if re.search(r"\b(שלום|היי|hello|hi)\b", low): return "greet", {}

    # הסכמה כללית (“מותר”, “מאשר”, “כן תריץ”, “go ahead”)
    if re.search(r"\b(מאשר|מסכים|מותר|go ahead|you can run|allow)\b", low):
        # אפשר לזהות גם קונטקסט (“בשביל לבנות”, “להריץ”)
        return "consent", {"purpose": "adapters/run", "ttl": 24*3600}

    # העדפות (“אני מעדיף עברית”, “תן מצב כהה”)
    if re.search(r"\b(עברית|hebrew|light mode|dark mode|ביצועים|בטיחות)\b", low):
        return "preference", {"text": m}

    # בקשת הסבר/מידע (grounded answer)
    if re.search(r"\b(למה|איך|תסביר|מה קורה|explain|why|what|answer)\b", low):
        # אוסף URLs/נתיבים אם יש; אם אין — נבקש בעצמנו בשכבת השיחה
        urls=re.findall(r"https?://\S+", m)
        files=re.findall(r"(?:^|\s)(/[^ \t\r\n]+)", m)
        return "ask_info", {"prompt": m, "urls": urls, "files":[p.strip() for p in files]}

    # “בנה לי אפליקציה / אתר / כלי”
    if re.search(r"\b(בנה|להקים|ליצור|build|create|make)\b.*\b(אפליקציה|אתר|כלי|app|site|tool)\b", low):
        spec=_json_in(m)
        return "build_app", {"free_text": m, "spec": spec}

    # “תריץ/תפעיל משהו” — גם בלי kind מפורש
    if re.search(r"\b(תריץ|תפעיל|run|execute|בדוק|בדיקה)\b", low):
        return "run_action", {"free_text": m, "params": _json_in(m)}

    return "unknown", {}
