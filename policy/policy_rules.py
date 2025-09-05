# policy/policy_rules.py
from __future__ import annotations
import time
import hashlib
from dataclasses import dataclass
from typing import Dict, Optional, Literal, Tuple

TrustLevel = Literal["low", "medium", "high", "system"]

@dataclass(frozen=True)
class UserPolicy:
    user_id: str
    trust_level: TrustLevel
    ttl_seconds: int
    require_evidence: bool
    require_strong_sources: bool
    require_freshness_seconds: Optional[int]
    max_claims_per_response: int
    max_ops_per_request: int
    enforce_user_subspace: bool
    allow_external_net: bool

DEFAULT_POLICIES: Dict[TrustLevel, UserPolicy] = {
    "low": UserPolicy(
        user_id="*",
        trust_level="low",
        ttl_seconds=24 * 3600,
        require_evidence=True,
        require_strong_sources=True,
        require_freshness_seconds=24 * 3600,
        max_claims_per_response=10,
        max_ops_per_request=128,
        enforce_user_subspace=True,
        allow_external_net=False,
    ),
    "medium": UserPolicy(
        user_id="*",
        trust_level="medium",
        ttl_seconds=7 * 24 * 3600,
        require_evidence=True,
        require_strong_sources=True,
        require_freshness_seconds=7 * 24 * 3600,
        max_claims_per_response=50,
        max_ops_per_request=512,
        enforce_user_subspace=True,
        allow_external_net=True,
    ),
    "high": UserPolicy(
        user_id="*",
        trust_level="high",
        ttl_seconds=30 * 24 * 3600,
        require_evidence=True,
        require_strong_sources=True,
        require_freshness_seconds=30 * 24 * 3600,
        max_claims_per_response=200,
        max_ops_per_request=2000,
        enforce_user_subspace=True,
        allow_external_net=True,
    ),
    "system": UserPolicy(
        user_id="*",
        trust_level="system",
        ttl_seconds=365 * 24 * 3600,
        require_evidence=True,
        require_strong_sources=True,
        require_freshness_seconds=None,  # system datasets pin versions instead
        max_claims_per_response=1000,
        max_ops_per_request=10000,
        enforce_user_subspace=False,
        allow_external_net=True,
    ),
}

class PolicyRegistry:
    def __init__(self):
        self._per_user: Dict[str, UserPolicy] = {}
    def set_user_policy(self, user_id: str, policy: UserPolicy) -> None:
        self._per_user[user_id] = policy
    def get_policy(self, user_id: str, default_level: TrustLevel="medium") -> UserPolicy:
        return self._per_user.get(user_id, DEFAULT_POLICIES[default_level])

# -------- Provenance & TTL enforcement --------

@dataclass(frozen=True)
class Evidence:
    uri: str                  # source URL or immutable content-addressed URI
    content: bytes            # raw bytes (or canonical serialized)
    fetched_at: float         # epoch seconds
    trust_tag: TrustLevel     # trust classification of the source
    signature: Optional[bytes] = None  # optional, if source provides
    mime: Optional[str] = None

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()

@dataclass
class EvidenceGate:
    # configurable thresholds that can also be policy-scoped
    min_trust: Dict[TrustLevel, int] = None
    def __post_init__(self):
        if self.min_trust is None:
            # rank trust -> numeric
            self.min_trust = {"low": 10, "medium": 50, "high": 80, "system": 100}

    def is_fresh(self, ev: Evidence, policy: UserPolicy) -> bool:
        if policy.require_freshness_seconds is None:
            return True
        return (time.time() - ev.fetched_at) <= policy.require_freshness_seconds

    def trust_score(self, ev: Evidence) -> int:
        # deterministic mapping; you can swap to a real catalog later
        return {"low": 25, "medium": 60, "high": 85, "system": 100}[ev.trust_tag]

    def check(self, ev: Evidence, policy: UserPolicy) -> Tuple[bool, str]:
        if policy.require_strong_sources:
            score = self.trust_score(ev)
            if score < self.min_trust.get(policy.trust_level, 50):
                return False, f"insufficient_trust: {score} < min_for_{policy.trust_level}"
        if policy.require_freshness_seconds is not None and not self.is_fresh(ev, policy):
            return False, "stale_evidence"
        return True, "ok"

class ProvenanceStore:
    """
    Content-addressable store: key=sha256(content), value={content, uri, fetched_at, trust}
    """
    def __init__(self):
        self._by_hash: Dict[str, Evidence] = {}
        self._by_uri: Dict[str, str] = {}  # uri -> sha256

    def put(self, ev: Evidence) -> str:
        h = ev.sha256
        self._by_hash[h] = ev
        self._by_uri[ev.uri] = h
        return h

    def get_by_hash(self, h: str) -> Optional[Evidence]:
        return self._by_hash.get(h)

    def get_by_uri(self, uri: str) -> Optional[Evidence]:
        h = self._by_uri.get(uri)
        return self._by_hash.get(h) if h else None

# -------- TTL ledger (per-user subspace) --------

class TTLIndex:
    """
    Tracks per-user object lifetimes (claims, memories, artifacts).
    """
    def __init__(self):
        self._ttl: Dict[str, float] = {}  # key -> expire_at

    def register(self, key: str, ttl_seconds: int) -> None:
        self._ttl[key] = time.time() + ttl_seconds

    def is_alive(self, key: str) -> bool:
        exp = self._ttl.get(key)
        return exp is not None and exp >= time.time()

    def purge_expired(self) -> int:
        now = time.time()
        to_del = [k for k, t in self._ttl.items() if t < now]
        for k in to_del:
            del self._ttl[k]
        return len(to_del)