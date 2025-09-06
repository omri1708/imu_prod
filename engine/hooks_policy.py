# engine/hooks_policy.py
# -*- coding: utf-8 -*-
"""
Hard gates for policy & provenance before execute/respond.
"""
from __future__ import annotations
from typing import List, Dict
from policy.policy_engine import policy_store, Budget, enforce_fs, enforce_net, PolicyViolation
from provenance.signing import Evidence, verify_evidence

def require_evidence(claims: List[Dict]) -> None:
    if not claims: raise PolicyViolation("claims required")
    for c in claims:
        ev = Evidence(**c["evidence"])
        if not verify_evidence(ev):
            raise PolicyViolation("bad evidence signature")
        # minimal freshness check (example)
        if ev.trust in ("low","unknown"):
            raise PolicyViolation("low-trust evidence rejected")

def before_io(user_id: str, path: str, is_write: bool):
    u = policy_store.get(user_id)
    enforce_fs(u, path, is_write)

def before_net(user_id: str, host: str, port: int):
    u = policy_store.get(user_id)
    enforce_net(u, host, port)

def new_budget(user_id: str) -> Budget:
    u = policy_store.get(user_id)
    return Budget(u.limits)