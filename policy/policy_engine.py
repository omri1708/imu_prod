# imu_repo/policy/policy_engine.py
from __future__ import annotations
from typing import Dict, Any

DEFAULT_POLICY = {
    # רמות סיכון ומדיניות ברירת־מחדל
    # משמעות: ככל שסיכון גבוה יותר → ספי אמון גבוהים יותר, TTL קצר יותר, ודורש מקורות מגוונים.
    "risk_levels": {
        "low":    {"min_trust": 0.65, "max_ttl_s": 7*24*3600,  "min_sources": 1, "freshness_decay": 0.10},
        "medium": {"min_trust": 0.75, "max_ttl_s": 72*3600,    "min_sources": 2, "freshness_decay": 0.15},
        "high":   {"min_trust": 0.85, "max_ttl_s": 24*3600,    "min_sources": 3, "freshness_decay": 0.20},
        "prod":   {"min_trust": 0.90, "max_ttl_s": 6*3600,     "min_sources": 3, "freshness_decay": 0.25},
    },
    # התאמות פר־דומיין (יכול להיות UI/Realtime/Data/Model וכו')
    "domain_overrides": {
        "ui_public": {"risk": "medium"},
        "ui_admin":  {"risk": "high"},
        "payments":  {"risk": "prod"},
        "realtime":  {"risk": "high"},
        "default":   {"risk": "medium"},
    }
}

class PolicyEngine:
    def __init__(self, policy: Dict[str,Any] | None = None):
        self.policy = policy or DEFAULT_POLICY

    def resolve(self, domain: str | None, risk_hint: str | None) -> Dict[str,Any]:
        rl = self.policy["risk_levels"]
        if risk_hint and risk_hint in rl:
            return {"risk": risk_hint, **rl[risk_hint]}
        dom = self.policy["domain_overrides"].get(domain or "default", {"risk": "medium"})
        r = dom.get("risk", "medium")
        return {"risk": r, **rl.get(r, rl["medium"])}