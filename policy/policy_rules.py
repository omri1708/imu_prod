# policy/policy_rules.py
from __future__ import annotations
import time, re, fnmatch, threading
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Literal, Tuple, List


TrustLevel = Literal["low", "medium", "high", "system"]
TRUST_LEVELS = ("guest", "basic", "trusted", "high_trust", "rootlike")

@dataclass
class RateLimit:
    capacity: int
    refill_per_sec: float
    _tokens: float = field(default=0.0)
    _last: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def allow(self, amount: int = 1) -> bool:
        now = time.time()
        with self._lock:
            self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.refill_per_sec)
            self._last = now
            if self._tokens >= amount:
                self._tokens -= amount
                return True
            return False

@dataclass
class UserPolicy:
    user_id: str
    trust_level: str = "basic"
    # TTL per artifact/evidence type (seconds)
    ttl_seconds: Dict[str, int] = field(default_factory=lambda: {
        "evidence": 90 * 24 * 3600,   # 90d
        "artifact": 30 * 24 * 3600,   # 30d
        "ui_cache": 7 * 24 * 3600,    # 7d
        "log": 30 * 24 * 3600,
    })
    # p95 bounds (milliseconds) per pipeline stage
    p95_bounds_ms: Dict[str, int] = field(default_factory=lambda: {
        "plan": 1500,
        "generate": 3500,
        "test": 4000,
        "verify": 2500,
        "package": 2000,
        "respond": 1200,
        "adapter": 5000,
    })
    # רשת: allow/deny
    net_allowlist_regex: List[str] = field(default_factory=lambda: [
        r"^https://(api\.)?github\.com/.*$",
        r"^https://(registry\.)?npmjs\.org/.*$",
        r"^https://dl\.google\.com/.*$",
        r"^https://developer\.android\.com/.*$",
        r"^https://services\.gradle\.org/.*$",
        r"^https://unity3d\.com/.*$",
        r"^https://packages\.ubuntu\.com/.*$",
        r"^https://(archive|security)\.ubuntu\.com/.*$",
        r"^https://pypi\.org/.*$",
        r"^https://(objects|storage)\.cloud\.googleapis\.com/.*$",
        r"^https://(download|developer)\.nvidia\.com/.*$",
        r"^wss://.*$",  # WebSocket (ייבדק מול hosts מורשים)
    ])
    net_blocklist_regex: List[str] = field(default_factory=lambda: [
        r"^http://.*$",             # חסום HTTP לא מאובטח
        r"^https://.*\.(ru|kp)$",   # דוגמה לחסימת TLDs
    ])
    ws_allowed_hosts: List[str] = field(default_factory=lambda: ["localhost", "127.0.0.1"])
    # קבצים: אילו נתיבים מותר לקרוא/לכתוב (דוגמא שמרנית)
    file_whitelist_glob: List[str] = field(default_factory=lambda: [
        "./workspace/**",
        "./.imu/**",
        "./artifacts/**",
        "./ui/**",
        "./adapters/**",
        "./tests/**",
    ])
    file_deny_glob: List[str] = field(default_factory=lambda: [
        "/etc/**", "C:\\Windows\\**", "/var/lib/**", "/root/**", "/home/*/.ssh/**"
    ])
    # מגבלות קצב לפי נושא (topic)
    rate_limits: Dict[str, RateLimit] = field(default_factory=lambda: {
        "telemetry": RateLimit(capacity=100, refill_per_sec=30),
        "logs": RateLimit(capacity=200, refill_per_sec=60),
        "ui_push": RateLimit(capacity=60, refill_per_sec=20),
        "build": RateLimit(capacity=10, refill_per_sec=0.1),  # build כבד—האטה
    })
    max_concurrent_streams: int = 16
    max_burst_global: int = 512

class PolicyEngine:
    """אכיפה שמרנית טרם פעולה: רשת/קבצים/קצב/WS/TTL/p95."""
    def __init__(self):
        self.users: Dict[str, UserPolicy] = {}
        self.global_burst = 0
        self._burst_lock = threading.Lock()
        self._burst_decay_last = time.time()

    def get(self, user_id: str) -> UserPolicy:
        if user_id not in self.users:
            self.users[user_id] = UserPolicy(user_id=user_id)
        return self.users[user_id]

    # --- Burst control (N*burst) ---
    def _decay_burst(self):
        now = time.time()
        with self._burst_lock:
            # דעיכה ליניארית פשוטה—מורידה 100 יחידות לשנייה
            self.global_burst = max(0, self.global_burst - int((now - self._burst_decay_last) * 100))
            self._burst_decay_last = now

    def try_burst(self, amount: int = 1) -> bool:
        self._decay_burst()
        with self._burst_lock:
            if self.global_burst + amount > 4096:  # תקרה מוחלטת
                return False
            self.global_burst += amount
            return True

    # --- Network rules ---
    def allow_url(self, user_id: str, url: str) -> bool:
        pol = self.get(user_id)
        if any(re.match(rx, url) for rx in pol.net_blocklist_regex):
            return False
        return any(re.match(rx, url) for rx in pol.net_allowlist_regex)

    def allow_ws_host(self, user_id: str, host: str) -> bool:
        pol = self.get(user_id)
        return host in pol.ws_allowed_hosts

    # --- File rules ---
    def allow_path(self, user_id: str, path: str, write: bool = False) -> bool:
        pol = self.get(user_id)
        path = path.replace("\\", "/")
        if any(fnmatch.fnmatch(path, g) for g in pol.file_deny_glob):
            return False
        if any(fnmatch.fnmatch(path, g) for g in pol.file_whitelist_glob):
            return True
        return False

    # --- Rate limits ---
    def rate_allow(self, user_id: str, topic: str, n: int = 1) -> bool:
        pol = self.get(user_id)
        rl = pol.rate_limits.get(topic)
        if not rl:
            return True
        return rl.allow(n)

    # --- p95 bounds ---
    def within_p95(self, user_id: str, stage: str, ms: int) -> bool:
        pol = self.get(user_id)
        bound = pol.p95_bounds_ms.get(stage)
        if bound is None:
            return True
        return ms <= bound

POLICY = PolicyEngine()

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