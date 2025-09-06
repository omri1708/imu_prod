# engine/adapter_runner.py
from __future__ import annotations
from typing import Dict, Any
from policy.user_subspace import registry
from storage.provenance_store import prov
from audit.audit_log import audit

from engine.policy import AskAndProceedPolicy, RequestContext
from adapters.android import AndroidAdapter
from adapters.ios import IOSAdapter
from adapters.unity import UnityAdapter
from adapters.cuda import CUDAAdapter
from adapters.k8s import K8sAdapter
from streaming.broker import StreamBroker


class ResourceRequired(Exception):
    def __init__(self, required: Dict[str,Any]): self.required=required


ADAPTERS = {
    "android": AndroidAdapter,
    "ios": IOSAdapter,
    "unity": UnityAdapter,
    "cuda": CUDAAdapter,
    "k8s": K8sAdapter,
}

class AdaptersService:
    def __init__(self, policy: AskAndProceedPolicy, broker: StreamBroker):
        self.policy = policy
        self.broker = broker

    def dry_run(self, adapter: str, spec: Dict[str, Any], ctx: RequestContext):
        cls = ADAPTERS.get(adapter)
        if not cls: raise ValueError(f"unknown_adapter:{adapter}")
        inst = cls(self.policy)
        plan = inst.plan(spec, ctx)
        inst.validate(plan, ctx)
        self.broker.publish("timeline", {
            "kind":"adapter_dry_run",
            "adapter":adapter,
            "commands":plan.commands,
            "env":plan.env,
            "notes":plan.notes,
            "user":ctx.user.user_id
        })
        return plan

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