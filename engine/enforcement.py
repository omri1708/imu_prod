# engine/enforcement.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, List
from policy.policy_engine import PolicyStore, PolicyViolation
from provenance.store import CAStore
from perf.measure import PerfRegistry

class EvidenceError(Exception): pass

class Enforcement:
    """
    שכבת אכיפה מרכזית: TTL/Trust/Evidence+p95.
    משולב בפייפליין לפני RESPOND ולפני rollout.
    """
    def __init__(self, policies:PolicyStore, ca:CAStore, perf:PerfRegistry):
        self.policies = policies
        self.ca = ca
        self.perf = perf

    def require_response_ok(self, user_id:str, claims:List[Dict[str,Any]], perf_key:str):
        p = self.policies.get(user_id)
        if not p:
            raise PolicyViolation("no user policy")

        # p95 guard
        p95 = self.perf.summary().get(perf_key, {}).get("p95_ms", 0.0)
        if p95 and p95 > p.max_p95_ms:
            raise PolicyViolation(f"p95 too high: {p95:.1f}ms > {p.max_p95_ms}ms")

        # evidence guard
        if p.require_evidence:
            if not claims:
                raise EvidenceError("require_evidence=True but no claims provided")
            for cl in claims:
                cid = cl.get("evidence_cid")
                if not cid:
                    raise EvidenceError("claim missing evidence_cid")
                if not self.ca.is_fresh_and_trusted(cid, p.ttl_ms, int(p.min_trust)):
                    raise EvidenceError("evidence not fresh/trusted")

    def require_cap_allowed(self, user_id:str, cap_id:str):
        p = self.policies.get(user_id)
        if not p or not p.allow_cap(cap_id):
            raise PolicyViolation(f"cap not allowed: {cap_id}")