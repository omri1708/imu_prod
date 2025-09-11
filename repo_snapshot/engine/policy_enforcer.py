# engine/policy_enforcer.py (אכיפת Grounding/TTL/Rate/Prio/P95)
# -*- coding: utf-8 -*-
from __future__ import annotations
import time, queue, threading
from typing import Dict, Any, Optional, Callable
from policy.policies import UserPolicy
from policy.ttl import enforce_ttl
from provenance.audit import AuditLog
from provenance.store import CAS
from grounded.http_verifier import HttpVerifier

class BackPressure:
    def __init__(self, limits_per_topic: Dict[str,float], priorities: Dict[str,int]):
        self.queues: Dict[str, queue.Queue] = {}
        self.limits = limits_per_topic
        self.priorities = priorities
        self.tokens: Dict[str, float] = {k:v for k,v in limits_per_topic.items()}  # tokens available per sec
        self.last_refill = time.time()

    def submit(self, topic: str, item: Dict[str,Any]):
        if topic not in self.queues:
            self.queues[topic] = queue.Queue(maxsize=1000)
        self.queues[topic].put(item, block=True)

    def _refill(self):
        now = time.time()
        dt = now - self.last_refill
        self.last_refill = now
        for t, rate in self.limits.items():
            self.tokens[t] = min(self.tokens.get(t, 0) + dt*rate, rate*2)  # burst cap = 2x rate

    def pump(self, handler: Callable[[str, Dict[str,Any]], None]):
        while True:
            self._refill()
            # pick next topic by priority that has tokens and items
            selected = None
            best_prio = 1e9
            for t, q in self.queues.items():
                if q.empty(): continue
                if self.tokens.get(t,0) <= 0: continue
                pr = self.priorities.get(t, 100)
                if pr < best_prio:
                    best_prio = pr
                    selected = t
            if not selected:
                time.sleep(0.005)
                continue
            self.tokens[selected] -= 1.0
            item = self.queues[selected].get()
            handler(selected, item)

class PolicyEnforcer:
    def __init__(self, user_policy: UserPolicy, cas: CAS, audit: AuditLog):
        self.user_policy = user_policy
        self.cas = cas
        self.audit = audit
        self.httpv = HttpVerifier(cas)
        self.bp = BackPressure(user_policy.rate_limits or {}, user_policy.priorities or {})

    def start_pump(self, handler: Callable[[str, Dict[str,Any]], None]):
        t = threading.Thread(target=self.bp.pump, args=(handler,), daemon=True)
        t.start()
        return t

    def grounded_guard(self, claims: Optional[list], route: str, start_ts: float):
        # Enforce P95 and Grounding
        if self.user_policy.require_grounding:
            if not claims or len(claims) == 0:
                raise RuntimeError("grounding_required: no claims supplied")
            for cl in claims:
                res = self.httpv.verify_claim(self.user_policy, cl)
                if not res["ok"]:
                    raise RuntimeError(f"grounding_failed: {res['reason']}")
                # attach evidence to CAS already in httpv
                self.audit.append(actor="engine", action="grounded_ok",
                                  payload={"route":route, "evidence":res["evidence_digest"]})
        budget = (self.user_policy.p95_budgets_ms or {}).get(route)
        if budget:
            elapsed_ms = (time.time() - start_ts)*1000.0
            if elapsed_ms > budget:
                raise RuntimeError(f"p95_budget_exceeded route={route} elapsed_ms={elapsed_ms:.1f} budget_ms={budget}")

    def enforce_ttl(self, index):
        removed = enforce_ttl(index, self.user_policy, time.time())
        if sum(removed.values())>0:
            self.audit.append("engine","ttl_cleanup",removed)
        return removed

    def submit_stream(self, topic: str, event: Dict[str,Any]):
        self.bp.submit(topic, event)