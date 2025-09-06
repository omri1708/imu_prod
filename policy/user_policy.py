# policy/user_policy.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Literal
import re


TRUST_TIERS = ("low", "medium", "high", "system")

@dataclass
class RateLimits:
    max_requests_per_min: int = 120
    burst: int = 30
    p95_latency_ms_ceiling: int = 2500

@dataclass
class NetworkPolicy:
    outbound_allowlist: List[str] = field(default_factory=lambda: [])
    outbound_blocklist: List[str] = field(default_factory=lambda: [])
    max_concurrent_sockets: int = 64
    per_topic_qps: Dict[str, float] = field(default_factory=dict)
    per_topic_burst: Dict[str, int] = field(default_factory=dict)
    server_side_throttle: Dict[str, Dict[str, float]] = field(default_factory=dict) # topic -> {qps, burst}

    def is_host_allowed(self, host: str) -> bool:
        if any(re.fullmatch(pat, host) for pat in self.outbound_blocklist):
            return False
        if not self.outbound_allowlist:
            return True
        return any(re.fullmatch(pat, host) for pat in self.outbound_allowlist)

@dataclass
class FilePolicy:
    allow_paths: List[str] = field(default_factory=lambda: [])
    max_file_mb: int = 64
    readonly_paths: List[str] = field(default_factory=lambda: [])

    def is_path_allowed(self, path: str, write: bool) -> bool:
        ok = any(path.startswith(p) for p in self.allow_paths) if self.allow_paths else True
        if not ok:
            return False
        if write and any(path.startswith(p) for p in self.readonly_paths):
            return False
        return True

@dataclass
class TTLPolicy:
    default_ttl_s: int = 60 * 60 * 24 * 30
    evidence_ttl_s_by_trust: Dict[str, int] = field(
        default_factory=lambda: {"low": 7*24*3600, "medium": 30*24*3600, "high": 180*24*3600, "system": 365*24*3600}
    )

@dataclass
class UserPolicy:
    user_id: str
    trust: str = "medium"
    ttl: TTLPolicy = field(default_factory=TTLPolicy)
    net: NetworkPolicy = field(default_factory=NetworkPolicy)
    files: FilePolicy = field(default_factory=FilePolicy)
    rate: RateLimits = field(default_factory=RateLimits)
    strict_grounding: bool = True
    require_signed_evidence: bool = True
    require_freshness_seconds: int = 30 * 24 * 3600  # 30 days
    per_capability_enable: Dict[str, bool] = field(default_factory=dict)

    def evidence_ttl(self) -> int:
        return self.ttl.evidence_ttl_s_by_trust.get(self.trust, self.ttl.default_ttl_s)

    def can_use_capability(self, name: str) -> bool:
        if name in self.per_capability_enable:
            return self.per_capability_enable[name]
        return True


##-------
Trust = Literal["low","medium","high","system"]
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

@dataclass
class TTLRule:
    seconds: int
    purge_on_revoke: bool = True


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