# policy/user_policy.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import time

Trust = float  # 0.0..1.0

@dataclass(frozen=True)
class UserSubspacePolicy:
    user_id: str
    min_trust_for_claim: Trust = 0.75         # דרישת אמון מינימלית לאישור claim
    max_ttl_seconds: int = 30 * 24 * 3600     # TTL דיפולטי לראיות (30 יום)
    strict_provenance: bool = True            # שרשרת יוחסין קשיחה חובה
    require_evidence_before_respond: bool = True
    # הגנות ריסון
    topic_rate_limits: Dict[str, int] = None  # msgs/second per topic
    burst_limit_global: int = 200             # מניעת N*burst גלובלי
    priority_overrides: Dict[str, int] = None # עדיפות לנושאים: נמוך=10, גבוה=1

    def with_overrides(self, **kw) -> "UserSubspacePolicy":
        d = self.__dict__.copy()
        d.update(kw)
        return UserSubspacePolicy(**d)

DEFAULT_POLICY = UserSubspacePolicy(
    user_id="default",
    topic_rate_limits={"telemetry": 50, "logs": 20, "timeline": 100, "progress": 200, "artifacts": 5},
    priority_overrides={"telemetry": 3, "timeline": 2, "progress": 1, "logs": 5, "artifacts": 4},
)

class PolicyRegistry:
    def __init__(self):
        self._by_user: Dict[str, UserSubspacePolicy] = {"default": DEFAULT_POLICY}

    def set(self, p: UserSubspacePolicy):
        self._by_user[p.user_id] = p

    def get(self, user_id: Optional[str]) -> UserSubspacePolicy:
        return self._by_user.get(user_id or "default", DEFAULT_POLICY)

POLICIES = PolicyRegistry()