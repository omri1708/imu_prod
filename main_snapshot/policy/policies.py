# policy/policies.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List, Literal
import time

Trust = Literal["system","org","team","user","external"]
Retention = Literal["session","short","standard","long","archival"]
Visibility = Literal["private","shared","public"]

@dataclass(frozen=True)
class UserPolicy:
    user_id: str
    # TTL in seconds per evidence/triple kind
    ttl_seconds: Dict[str, int]
    # minimal trust per source kind (e.g. 'web','repo','signed','sensor')
    min_trust: Dict[str, Trust]
    # max staleness in seconds per domain (e.g. 'finance','weather','docs')
    max_staleness: Dict[str, int]
    # who can see artifacts
    visibility: Visibility = "private"
    # strict grounding requirement
    require_grounding: bool = True
    # require provenance signature for high-impact actions
    require_signature_for: List[str] = None
    # p95 latency budgets per route (ms)
    p95_budgets_ms: Dict[str, int] = None
    # rate limits per topic (events/sec)
    rate_limits: Dict[str, float] = None
    # priority classes
    priorities: Dict[str, int] = None  # lower = higher priority

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["require_signature_for"] = self.require_signature_for or []
        d["p95_budgets_ms"] = self.p95_budgets_ms or {}
        d["rate_limits"] = self.rate_limits or {}
        d["priorities"] = self.priorities or {}
        return d

DEFAULT_POLICY = UserPolicy(
    user_id="*",
    ttl_seconds={
        "claim": 7*24*3600,
        "evidence": 30*24*3600,
        "artifact": 90*24*3600,
        "session": 24*3600,
    },
    min_trust={
        "web": "external",
        "repo": "org",
        "signed": "team",
        "sensor": "team",
    },
    max_staleness={
        "weather": 3*3600,
        "finance": 60,
        "docs": 365*24*3600,
        "code": 365*24*3600,
    },
    visibility="private",
    require_grounding=True,
    require_signature_for=["deploy","publish","pay","delete"],
    p95_budgets_ms={"respond": 2500, "plan": 1500, "verify": 1200, "deploy": 10000},
    rate_limits={"telemetry": 200.0, "logs": 100.0, "timeline": 50.0},
    priorities={"logic": 0, "telemetry": 1, "logs": 2}
)

class PolicyRegistry:
    def __init__(self):
        self._by_user: Dict[str, UserPolicy] = {"*": DEFAULT_POLICY}

    def get(self, user_id: str) -> UserPolicy:
        return self._by_user.get(user_id, self._by_user["*"])

    def put(self, policy: UserPolicy):
        self._by_user[policy.user_id] = policy

POLICIES = PolicyRegistry()