# engine/enforcer.py (אכיפת Evidence/TTL/Trust לפני תגובה)
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Any
from policy.user_policy import POLICIES
from provenance.cas import is_valid

class GroundingError(Exception): pass

def enforce_claims(user_id: str, claims: List[Dict[str, Any]]):
    """
    claims – רשימת dict עם: {"hash": <sha256>, "about": "…", "trust_hint": float?}
    נדרש: כל claim חייב להיות עם evidence תקף ב-CAS לפי מדיניות המשתמש.
    """
    pol = POLICIES.get(user_id)
    if not pol.require_evidence_before_respond:
        return
    if not claims:
        raise GroundingError("Claims required before respond (policy requires evidence).")
    # כל claim חייב להיות תקף
    for c in claims:
        h = c.get("hash")
        if not h or not is_valid(h, pol.min_trust_for_claim):
            raise GroundingError(f"Invalid/expired/low-trust evidence for claim: {c!r}")