# engine/contracts_gate.py
# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Optional
from governance.policy import RespondPolicy
from grounded.evidence_contracts import EvidenceIndex, Evidence, compute_sha256
from contracts.errors import ContractViolation


class ContractViolation(Exception): ...
class PolicyDenied(Exception): ...

def enforce_respond_contract(text: str,
                             claims: Optional[List[Dict[str,Any]]],
                             evidence: Optional[List[Dict[str,Any]]],
                             policy: RespondPolicy,
                             ev_index: EvidenceIndex) -> None:
    """אוכף מדיניות + התאמת ראיות לאינדקס (sha/ts/trust/sig/url)."""
    policy.enforce(text=text, claims=claims, evidence=evidence)
    if evidence:
        for evd in evidence:
            ev = Evidence(sha256=evd["sha256"], ts=int(evd["ts"]), trust=float(evd["trust"]),
                          url=str(evd["url"]), sig_ok=bool(evd["sig_ok"]))
            ev_index.verify(ev)
    
    if not text or not isinstance(text, str):
        raise ContractViolation("empty_text")

    if _is_math_expression(text):
        if policy.allow_math_without_claims:
            return
        # אם לא מותר – חייבים claims/evidence גם לחשבון
    if policy.require_claims and not claims:
        raise PolicyDenied("claims_required")

    if len(claims or []) > int(policy.max_claims):
        raise PolicyDenied("too_many_claims")

    if policy.require_evidence:
        if not evidence:
            raise PolicyDenied("evidence_required")
        # כל עדות חייבת לעמוד בכלל
        for e in evidence:
            sha = str(e.get("sha256",""))
            if not sha or not ev_index.validate(sha, policy.evidence):
                raise PolicyDenied(f"evidence_invalid:{sha}")


def attach_claim(text: str, source_bytes: bytes, url: str, trust: float, sig_ok: bool, ev_index: EvidenceIndex) -> Dict[str,Any]:
    sha = compute_sha256(source_bytes)
    rec = {"sha256":sha,"ts":0,"trust":trust,"url":url,"sig_ok":sig_ok}
    ev_index.put(sha, {"ts":0,"trust":trust,"url":url,"sig_ok":sig_ok})
    return rec


def _is_math_expression(text: str) -> bool:
    # זיהוי מאוד שמרני (ללא צד שלישי) – ביטוי מספרי פשוט
    import re
    return bool(re.fullmatch(r"[0-9\.\+\-\*\/\(\) \t]+", text or ""))


 