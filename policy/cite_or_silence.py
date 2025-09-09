# policy/cite_or_silence.py
from __future__ import annotations
from typing import Any, Dict

REQUIRED_KEYS = ("sources", "citations", "evidence")


def require_citations_or_block(text: str, *, meta: Dict[str,Any], policy: Dict[str,Any]) -> Dict[str,Any]:
    """אם אין מקורות/ציטוטים — אין תשובה. מחזיר מבנה אחיד עם סיבת חסימה והצעות פעולה."""
    ev = meta or {}
    has_cite = any(k in ev and ev[k] for k in REQUIRED_KEYS)
    if has_cite:
        return {"ok": True, "text": text, "meta": {k: ev.get(k) for k in REQUIRED_KEYS}}
    # חסימה שקופה עם דרך פעולה
    return {
        "ok": False,
        "error": "no_citations",
        "action": {
            "ask_user": "נאשר חיפוש מקורות אמינים? או לצמצם את השאלה לפרטים שכבר נמצאים בזיכרון/מסמכים פנימיים.",
            "options": ["search_web", "use_internal_sources", "rephrase"],
        }
    }