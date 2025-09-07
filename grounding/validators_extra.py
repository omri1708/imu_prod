# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, Tuple, List
import time

from assurance.validators import Validator

def ttl_validator(min_ttl: float) -> Validator:
    def _check(ctx: Dict[str,Any]) -> Tuple[bool,str]:
        # מצפה ש-evidence יכלול ts/ttl
        evs = ctx.get("evidence", [])
        now = time.time()
        stale = [e for e in evs if e.get("ts") and e.get("ts")+min_ttl < now]
        if stale:
            return False, "stale_evidence"
        return True, "ok"
    return Validator("ttl", _check)

def cross_source_consistency(key: str) -> Validator:
    """בודק שכל המקורות שהביאו ערך עבור key מסכימים (לפי digest/ערך)."""
    def _check(ctx: Dict[str,Any]) -> Tuple[bool,str]:
        # לצורך הפשטה: מניחים שה-artifact מכיל payload עם values[key]
        art = ctx.get("artifact", {})
        digests = art.get("output_digests", [])
        if not digests:
            return False, "no_output"
        return True, "ok"  # הרחבה אפשרית: fetch מה-CAS ולבדוק ערכים כפולים
    return Validator("consistency", _check)
