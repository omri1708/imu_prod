from __future__ import annotations
import os
from typing import Dict, Any

def vertex_guard(prompt: str, *, allow_live: bool) -> Dict[str,Any]:
    # DRYRUN אם אין מפתחות/אישור
    if not allow_live or not os.environ.get("VERTEX_PROJECT"):
        return {"ok": True, "mode": "dryrun", "request": {"text": prompt}}
    # כאן תוסיף קריאה אמיתית ל-Vertex AI לפי ה-SDK/REST אצלך
    return {"ok": True, "mode": "live", "request": {"text": prompt}}
