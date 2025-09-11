# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any

def evidence_gate(art: Dict[str, Any]) -> Dict[str, Any]:
    # דוגמה: ודא שיש בדיקות, ציטוטים נדרשים, ושטריגרים לא עוברים מוגדרים
    ok = True; reasons = []
    if not art.get("tests"): ok=False; reasons.append("missing tests")
    if art.get("requires_sources") and not art.get("citations"): ok=False; reasons.append("missing citations")
    return {"ok": ok, "reasons": reasons}
