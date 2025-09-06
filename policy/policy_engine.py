# imu_repo/policy/policy_engine.py
from __future__ import annotations
from typing import Dict, Any
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Literal
import time, json, hashlib
from typing import Dict, Any, Optional, Tuple

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


Trust = Literal["unknown","low","medium","high","pinned"]
Decision = Literal["allow","block","require_consent"]



class PolicyViolation(Exception):
    pass

class TrustLevel:
    # 0=לא אמין, 1=נמוך, 2=בינוני, 3=גבוה, 4=מוסמך
    def __init__(self, level:int):
        assert 0 <= level <= 4
        self.level = level
    def __int__(self): return self.level
    def __repr__(self): return f"TrustLevel({self.level})"

def now_ms() -> int:
    return int(time.time() * 1000)

class UserPolicy:
    """
    מדיניות קשיחה פר-משתמש:
     - ttl_ms: חיי נתון/ראייה (פג-תוקף)
     - min_trust: רמת אמון מינימלית למקור
     - max_p95_ms: סף p95 לביצועים (Pipeline יבדוק)
     - require_evidence: השבה מותרת רק אם יש Claims+Evidence מאומתת
     - sandbox_caps: אילו יכולות מותרות (רשימת מזהים)
    """
    def __init__(self, user_id:str, ttl_ms:int, min_trust:int,
                 max_p95_ms:int, require_evidence:bool, sandbox_caps:Tuple[str,...]):
        self.user_id = user_id
        self.ttl_ms = ttl_ms
        self.min_trust = TrustLevel(min_trust)
        self.max_p95_ms = max_p95_ms
        self.require_evidence = require_evidence
        self.sandbox_caps = set(sandbox_caps)

    def allow_cap(self, cap_id:str):
        return cap_id in self.sandbox_caps

    def to_dict(self) -> Dict[str,Any]:
        return {
            "user_id": self.user_id, "ttl_ms": self.ttl_ms,
            "min_trust": int(self.min_trust),
            "max_p95_ms": self.max_p95_ms,
            "require_evidence": self.require_evidence,
            "sandbox_caps": sorted(self.sandbox_caps)
        }

class PolicyStore:
    """אחסון מדיניות בזיכרון (אפשר להחליף ל-DB)."""
    def __init__(self):
        self._by_user: Dict[str,UserPolicy] = {}

    def set_policy(self, p:UserPolicy):
        self._by_user[p.user_id] = p

    def get(self, user_id:str) -> Optional[UserPolicy]:
        return self._by_user.get(user_id)

    def snapshot_hash(self) -> str:
        blob = json.dumps({u:p.to_dict() for u,p in sorted(self._by_user.items())}, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()


@dataclass
class PolicyRule:
    name: str
    topic: str                      # e.g. "adapter.unity.run" / "net.ws.publish"
    action: str                     # e.g. "invoke" / "read" / "write"
    decision: Decision             # allow | block | require_consent
    ttl_sec: Optional[int] = None   # per-user TTL for cached grants
    min_trust: Trust = "unknown"    # minimal provenance trust required
    max_rate_per_min: Optional[int] = None  # throttling
    priority: int = 100             # lower = higher priority

@dataclass
class CachedGrant:
    granted_at: float
    ttl_sec: int

@dataclass
class UserSubspacePolicy:
    user_id: str
    rules: List[PolicyRule] = field(default_factory=list)
    grants: Dict[str, CachedGrant] = field(default_factory=dict)  # key = topic:action

    def decide(self, topic: str, action: str, trust: Trust, rate_counter_per_min: int) -> Decision:
        now = time.time()
        key = f"{topic}:{action}"
        # TTL grant reuse
        if key in self.grants:
            g = self.grants[key]
            if now - g.granted_at <= g.ttl_sec:
                return "allow"
            else:
                self.grants.pop(key, None)

        matched = sorted(
            [r for r in self.rules if r.topic==topic and r.action==action],
            key=lambda r: r.priority
        )
        if not matched:
            return "require_consent"

        rule = matched[0]
        # trust gate
        order = ["unknown","low","medium","high","pinned"]
        if order.index(trust) < order.index(rule.min_trust):
            return "require_consent"

        # rate limiting
        if rule.max_rate_per_min is not None and rate_counter_per_min > rule.max_rate_per_min:
            return "block"

        if rule.decision == "allow":
            if rule.ttl_sec:
                self.grants[key] = CachedGrant(granted_at=now, ttl_sec=rule.ttl_sec)
            return "allow"

        return rule.decision

    def grant_once(self, topic: str, action: str, ttl_sec: int):
        self.grants[f"{topic}:{action}"] = CachedGrant(time.time(), ttl_sec)

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