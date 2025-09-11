# policy/enforce.py
from __future__ import annotations
from typing import List, Dict
from policy.policy_rules import UserPolicy, Evidence, EvidenceGate, ProvenanceStore, TTLIndex

class PolicyEnforcer:
    def __init__(self, prov: ProvenanceStore, ttl: TTLIndex, gate: EvidenceGate):
        self.prov = prov
        self.ttl = ttl
        self.gate = gate

    def assert_claims(self, user_policy: UserPolicy, claims: List[Dict]) -> None:
        """
        Each claim must include an evidence list; we enforce:
        - count limit
        - evidence existence
        - trust/freshness via EvidenceGate
        - register TTL for claim handle
        """
        if len(claims) > user_policy.max_claims_per_response:
            raise ValueError("too_many_claims")

        for c in claims:
            evidences = c.get("evidence", [])
            if user_policy.require_evidence and not evidences:
                raise ValueError("evidence_required")

            ok_any = False
            for ev_in in evidences:
                # materialize Evidence
                ev = Evidence(
                    uri=ev_in["uri"],
                    content=ev_in["content"] if isinstance(ev_in["content"], (bytes, bytearray)) else ev_in["content"].encode("utf-8"),
                    fetched_at=ev_in.get("fetched_at", 0.0),
                    trust_tag=ev_in.get("trust_tag", user_policy.trust_level),
                    signature=ev_in.get("signature"),
                    mime=ev_in.get("mime")
                )
                h = self.prov.put(ev)
                ok, why = self.gate.check(ev, user_policy)
                if ok:
                    ok_any = True
                else:
                    # allow multiple; at least one must pass
                    pass
                # TTL for the evidence blob
                self.ttl.register(f"evidence:{h}", user_policy.ttl_seconds)

            if user_policy.require_evidence and not ok_any:
                raise ValueError("no_strong_fresh_evidence")

            # TTL registration for the claim itself
            if "id" in c:
                self.ttl.register(f"claim:{c['id']}", user_policy.ttl_seconds)