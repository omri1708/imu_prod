# governance/user_policy.py (מדיניות קשיחה פר-משתמש)
# -*- coding: utf-8 -*-
import time
from typing import Dict, Tuple
from governance.policy import RespondPolicy, EvidenceRule
from grounded.evidence_contracts import EvidenceIndex

# מאגר מדיניות/אינדקסים לכל משתמש (תת-מרחב)
_USERS: Dict[str, Tuple[RespondPolicy, EvidenceIndex]] = {}

# פרופילי ברירת מחדל לדוגמה — אפשר להרחיב/לשנות דינמית
DEFAULTS = {
    "anonymous": EvidenceRule(min_trust=0.80, max_age_sec=3*24*3600, allowed_domains=["example.com"], require_signature=True),
    "power_user": EvidenceRule(min_trust=0.90, max_age_sec=24*3600, allowed_domains=["gov.il","who.int","iso.org","example.com"], require_signature=True),
    "strict_org": EvidenceRule(min_trust=0.95, max_age_sec=6*3600, allowed_domains=["corp.local","corp.example"], require_signature=True),
}

def ensure_user(user_id: str):
    if user_id not in _USERS:
        ev_rule = DEFAULTS.get(user_id, DEFAULTS["anonymous"])
        pol = RespondPolicy(
            require_claims=True,
            require_evidence=True,
            evidence=ev_rule,
            allow_math_without_claims=False,  # מחמיר: אין תשובה בלי Claims/Evidence
            max_claims=64
        )
        _USERS[user_id] = (pol, EvidenceIndex())

def get_user_policy(user_id: str) -> Tuple[RespondPolicy, EvidenceIndex]:
    ensure_user(user_id)
    return _USERS[user_id]

def set_user_policy(user_id: str, rule: EvidenceRule, allow_math_without_claims: bool = False, max_claims: int = 64):
    pol = RespondPolicy(require_claims=True, require_evidence=True, evidence=rule,
                        allow_math_without_claims=allow_math_without_claims, max_claims=max_claims)
    _USERS[user_id] = (pol, EvidenceIndex())

def tighten_ttl(user_id: str, max_age_sec: int):
    pol, ev = get_user_policy(user_id)
    pol.evidence.max_age_sec = int(max_age_sec)

def restrict_domains(user_id: str, domains):
    pol, ev = get_user_policy(user_id)
    pol.evidence.allowed_domains = list(domains)

def raise_trust_floor(user_id: str, min_trust: float):
    pol, ev = get_user_policy(user_id)
    pol.evidence.min_trust = float(min_trust)