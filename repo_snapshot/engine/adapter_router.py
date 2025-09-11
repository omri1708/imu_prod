# engine/adapter_router.py (חיבור API ← Stream Broker ← UI-DSL, כולל throttling לפי policy)
# -*- coding: utf-8 -*-
from __future__ import annotations
import json, time, threading
from typing import Dict, Any, Callable
from policy.policies import POLICIES
from provenance.audit import AuditLog
from provenance.store import CAS
from engine.policy_enforcer import PolicyEnforcer

class StreamBroker:
    def __init__(self, enforcer: PolicyEnforcer, publish: Callable[[str,Dict[str,Any]],None]):
        self.enforcer = enforcer
        self.publish = publish
        enforcer.start_pump(self._handle)

    def submit(self, topic: str, event: Dict[str,Any]):
        self.enforcer.submit_stream(topic, event)

    def _handle(self, topic: str, event: Dict[str,Any]):
        # server-side throttling already applied by enforcer; just forward
        self.publish(topic, event)

def default_publisher(topic: str, event: Dict[str,Any]):
    #TODO- placeholder transport to websocket hub you already wired (no-op here)
    pass

def new_broker(user_id: str, cas: CAS, audit: AuditLog) -> StreamBroker:
    policy = POLICIES.get(user_id)
    enforcer = PolicyEnforcer(policy, cas, audit)
    return StreamBroker(enforcer, default_publisher)