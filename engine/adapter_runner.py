# engine/adapter_runner.py
from __future__ import annotations
from typing import Dict, Any
from policy.user_subspace import registry
from storage.provenance_store import prov
from audit.audit_log import audit

class ResourceRequired(Exception):
    def __init__(self, required: Dict[str,Any]): self.required=required

def enforce_policy(user_id: str, topic: str, trust: str, rate_counter_min: int=0):
    pol = registry.ensure_user(user_id)
    decision = pol.decide(topic=topic, action="invoke", trust=trust, rate_counter_per_min=rate_counter_min)
    if decision == "allow":
        return
    if decision == "block":
        raise PermissionError(f"Blocked by policy: {topic}")
    raise ResourceRequired({"consent_for": topic, "policy": pol.user_id})

def record_artifact_bytes(b: bytes, uri: str|None, trust: str, meta=None) -> str:
    sha = prov.put_bytes(b, uri=uri, trust=trust, meta=meta)
    audit.append({"event":"artifact.put", "sha": sha, "uri": uri, "trust": trust})
    return sha