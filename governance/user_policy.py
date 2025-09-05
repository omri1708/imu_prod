from typing import Dict, Tuple, Any
import time
from governance.policy import RespondPolicy, EvidenceRule, Subspace
from grounded.evidence_contracts import EvidenceIndex


_USERS: Dict[str, Tuple[RespondPolicy, EvidenceIndex, Subspace]] = {}


DEFAULTS = {
    "anonymous": EvidenceRule(min_trust=0.80, max_age_sec=3*24*3600,
                              allowed_domains=["example.com","who.int"], require_signature=True),
    "power_user": EvidenceRule(min_trust=0.90, max_age_sec=24*3600,
                               allowed_domains=["gov.il","who.int","iso.org","example.com"], require_signature=True),
    "strict_org": EvidenceRule(min_trust=0.95, max_age_sec=6*3600,
                               allowed_domains=["corp.local","corp.example"], require_signature=True),
}

class Policy:
    def __init__(self, *, require_evidence: bool = True, trust_threshold: float = 0.7,
                 ttl_seconds: int = 90*24*3600, max_sleep_ms: int = 10_000):
        self.require_evidence = require_evidence
        self.trust_threshold  = trust_threshold
        self.ttl_seconds      = ttl_seconds
        self.max_sleep_ms     = max_sleep_ms

class EvidenceIndex:
    """
    נקודת חיבור לציוני אמון חיצוניים (חתימות/מקורות/עדכניות).
    כאן – פשטני; קל להחליף במימוש עשיר.
    """
    def score(self, ev) -> float:
        # אם יש hash ב-CAS – בסיס 0.8; אפשר לשקלל לפי provenance/חתימות/טריות
        return 0.8


def get_user_policy(user: str) -> Tuple[Policy, EvidenceIndex]:
    # דוגמה: פר־משתמש מחמירים (תת-מרחבים/TTL/ספי אמון)
    if user.startswith("root:"):
        return Policy(require_evidence=True, trust_threshold=0.9, ttl_seconds=7*24*3600), EvidenceIndex()
    return Policy(require_evidence=True, trust_threshold=0.7, ttl_seconds=90*24*3600), EvidenceIndex()


def ensure_user(user_id: str, role: str = "user", org: str = "public", project: str = "default"):
    if user_id not in _USERS:  # noqa: E201 (רווח מודגש למניעת בלבול)
        ev_rule = DEFAULTS.get(user_id, DEFAULTS["anonymous"])
        pol = RespondPolicy(require_claims=True, require_evidence=True,
                            evidence=ev_rule, allow_math_without_claims=False, max_claims=64)
        _USERS[user_id] = (pol, EvidenceIndex(), Subspace(role=role, org=org, project=project))


def get_user_subspace(user_id: str) -> Subspace:
    ensure_user(user_id)
    return _USERS[user_id][2]

def set_user_policy(user_id: str, rule: EvidenceRule, allow_math_without_claims: bool = False, max_claims: int = 64):
    ensure_user(user_id)
    _, ev, ss = _USERS[user_id]
    pol = RespondPolicy(require_claims=True, require_evidence=True, evidence=rule,
                        allow_math_without_claims=allow_math_without_claims, max_claims=max_claims)
    _USERS[user_id] = (pol, ev, ss)

def tighten_ttl(user_id: str, max_age_sec: int):
    pol, ev = get_user_policy(user_id)
    pol.evidence.max_age_sec = int(max_age_sec)

def restrict_domains(user_id: str, domains):
    pol, ev = get_user_policy(user_id)
    pol.evidence.allowed_domains = list(domains)

def raise_trust_floor(user_id: str, min_trust: float):
    pol, ev = get_user_policy(user_id)
    pol.evidence.min_trust = float(min_trust)