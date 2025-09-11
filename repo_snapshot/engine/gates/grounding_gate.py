# imu_repo/engine/gates/grounding_gate.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from grounded.evidence_store import EvidenceStore
from grounded.provenance import validate_claim

class GroundingGate:
    """
    מוודא שאין 'הלוצינציה מערכתית':
      - לכל claim יש לפחות עדות אחת
      - עדות מאומתת (sha/sig/fresh/domain)
      - ניתן לדרוש 'min_good_evidence' לכל claim (>=1)
      - ניתן להגדיר רשימת דומיינים מותרים (allowed_domains)
    קלט צפוי (bundle):
      {
        "text": "...",
        "claims": [
          {"id":"c1","statement":"X","evidence":["sha1","sha2"],"schema":{...}},
          ...
        ]
      }
    """
    def __init__(self, *,
                 allowed_domains: Optional[List[str]]=None,
                 require_signature: bool=True,
                 min_good_evidence: int=1):
        self.allowed_domains = allowed_domains or []
        self.require_signature = bool(require_signature)
        self.min_good = int(min_good_evidence)
        self.store = EvidenceStore()

    def check(self, bundle: Dict[str,Any], *, now: float | None=None) -> Dict[str,Any]:
        claims = bundle.get("claims") or []
        if not claims:
            return {"ok": False, "reason": "no_claims", "violations": [("no_claims",)]}
        viol=[]
        results=[]
        for cl in claims:
            r = validate_claim(cl, self.store,
                               allowed_domains=self.allowed_domains,
                               require_sig=self.require_signature,
                               now=now)
            results.append(r)
            good = sum(1 for e in r["evidence_results"] if e["ok"])
            if not r["ok"] or good < self.min_good:
                viol.append(("claim_failed", cl.get("id"), {"good":good, "need":self.min_good, "schema_ok": r["schema_ok"], "schema_reason": r["schema_reason"], "evidence": r["evidence_results"]}))
        return {"ok": len(viol)==0, "violations": viol, "results": results}