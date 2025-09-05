# policy/user_subspace.py
from __future__ import annotations
from typing import Dict
from .policy_engine import UserSubspacePolicy, PolicyRule

class UserPolicyRegistry:
    def __init__(self):
        self._by_user: Dict[str, UserSubspacePolicy] = {}

    def ensure_user(self, user_id: str) -> UserSubspacePolicy:
        if user_id not in self._by_user:
            self._by_user[user_id] = UserSubspacePolicy(
                user_id=user_id,
                rules=[
                    PolicyRule(name="ws.publish.safe", topic="net.ws.publish", action="invoke",
                               decision="allow", ttl_sec=3600, min_trust="medium", max_rate_per_min=600, priority=10),
                    PolicyRule(name="adapter.run.consent", topic="adapter.*.run", action="invoke",
                               decision="require_consent", ttl_sec=900, min_trust="low", priority=20),
                    PolicyRule(name="provenance.pin.read", topic="prov.read", action="read",
                               decision="allow", ttl_sec=3600, min_trust="unknown", priority=5),
                ]
            )
        return self._by_user[user_id]

# singleton
registry = UserPolicyRegistry()