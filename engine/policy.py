# engine/policy.py
# -*- coding: utf-8 -*-
from typing import List, Dict, Any
from provenance.store import CASStore, EvidenceMeta

class GroundingPolicy:
    """
    מחייב לפחות ראיה אחת מאומתת (verify_meta==True) ולערך trust>=threshold.
    """
    def __init__(self, trust_threshold: float = 0.6):
        self.trust_threshold = float(trust_threshold)
        self.cas = CASStore()

    def check(self, claims: List[Dict[str, Any]]) -> bool:
        if not claims:
            return False
        ok = False
        for c in claims:
            digest = c.get("sha256")
            if not digest:
                continue
            meta = self.cas.get(digest)
            if not meta:
                continue
            if meta.trust >= self.trust_threshold and self.cas.verify_meta(meta):
                ok = True
            else:
                return False
        return ok