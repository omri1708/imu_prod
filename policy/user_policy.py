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
    trust_floor: int = 60           # אי אפשר לצרוך ראיה מתחת לרף הזה
    require_signature: bool = False # אפשר להקשיח ל־True
    ttl_s_soft: int = 7*24*3600     # TTL “רך” להצעות/סקיצות
    ttl_s_hard: int = 30*24*3600    # TTL קשיח לראיות חתומות
    p95_budget_ms: int = 1200       # תקציב ביצועים
    deny_external_net: bool = False # אפשר לסגור לגמרי

DEFAULT_POLICY = UserSubspacePolicy(user_id="default")

class PolicyRegistry:
    def __init__(self):
        self._by_user: Dict[str, UserSubspacePolicy] = {"default": DEFAULT_POLICY}

    def set(self, p: UserSubspacePolicy):
        self._by_user[p.user_id] = p

    def get(self, user_id: Optional[str]) -> UserSubspacePolicy:
        return self._by_user.get(user_id or "default", DEFAULT_POLICY)

POLICIES = PolicyRegistry()