# imu_repo/engine/fallbacks.py
from __future__ import annotations
import time
from typing import Dict, Any
from grounded.claims import current

def safe_text_fallback(*, reason: str, details: Dict[str,Any] | None = None) -> str:
    """
    מחזיר תגובה בטוחה כאשר Guard חוסם.
    הוספת ראיה על שימוש ב-fallback, כדי לשמור שקיפות מלאה.
    """
    payload = {"reason": reason, "details": details or {}, "ts": time.time()}
    current().add_evidence("fallback_used", {
        "source_url": "local://fallback",
        "trust": 0.95,
        "ttl_s": 600,
        "payload": payload
    })
    # טקסט ברור + מתוייג — ניתן לסינון לוגים
    return f"[FALLBACK] Guard prevented direct response. reason={reason}; details={payload.get('details',{})}"