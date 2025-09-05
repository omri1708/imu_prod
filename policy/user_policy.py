# policy/user_policy.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Literal
import time

Trust = Literal["low","medium","high","system"]

@dataclass
class TTLRule:
    seconds: int
    purge_on_revoke: bool = True

@dataclass
class UserPolicy:
    user_id: str
    trust: Trust = "low"
    ttl_by_kind: Dict[str, TTLRule] = field(default_factory=lambda: {
        "evidence": TTLRule(90*24*3600),
        "log": TTLRule(30*24*3600),
        "artifact": TTLRule(180*24*3600),
        "profile": TTLRule(365*24*3600)
    })
    require_grounded_response: bool = True
    max_sleep_ms: int = 5_000
    max_ws_conns: int = 32
    throttle_per_topic_qps: float = 25.0
    throttle_burst: int = 100

@dataclass
class PolicyStore:
    _by_user: Dict[str,UserPolicy] = field(default_factory=dict)

    def get(self, user_id:str) -> UserPolicy:
        return self._by_user.setdefault(user_id, UserPolicy(user_id=user_id))

    def set_trust(self, user_id:str, trust:Trust):
        self.get(user_id).trust = trust

    def set_ttl(self, user_id:str, kind:str, seconds:int):
        p = self.get(user_id)
        p.ttl_by_kind[kind] = TTLRule(seconds)


POLICY = PolicyStore()


def ttl_for(user_id:str, kind:str) -> int:
    p = POLICY.get(user_id)
    if kind in p.ttl_by_kind: 
        return p.ttl_by_kind[kind].seconds
    return 30*24*3600

def enforce_sleep_ms(user_id:str, ms:int):
    p = POLICY.get(user_id)
    if ms > p.max_sleep_ms:
        raise RuntimeError(f"sleep_ms_exceeds_policy: requested={ms} > max={p.max_sleep_ms}")

def per_topic_limits(user_id:str):
    p = POLICY.get(user_id)
    return dict(qps=p.throttle_per_topic_qps, burst=p.throttle_burst)

def must_be_grounded(user_id:str) -> bool:
    return POLICY.get(user_id).require_grounded_response



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