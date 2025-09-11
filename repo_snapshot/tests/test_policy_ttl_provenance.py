# tests/test_policy_ttl_provenance.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, shutil
from policy.policy_engine import PolicyStore, UserPolicy
from provenance.store import CAStore
from engine.enforcement import Enforcement, EvidenceError
from perf.measure import PerfRegistry

def test_enforce_evidence_and_ttl():
    root = "./.ca_test"
    if os.path.exists(root): shutil.rmtree(root)
    ca = CAStore(root=root, secret_key=b"t")
    pol = PolicyStore()
    pol.set_policy(UserPolicy("u", ttl_ms=10_000, min_trust=3, max_p95_ms=999999, require_evidence=True, sandbox_caps=()))
    perf = PerfRegistry()
    enf = Enforcement(pol, ca, perf)

    cid = ca.put(b'{"k":"v"}', source="unit", trust=3)
    claims = [{"evidence_cid": cid}]
    perf.track("pipeline.total", 10.0)
    enf.require_response_ok("u", claims, "pipeline.total")  # לא יזרוק

def test_reject_on_low_trust():
    root = "./.ca_test2"
    if os.path.exists(root): shutil.rmtree(root)
    ca = CAStore(root=root, secret_key=b"t2")
    pol = PolicyStore()
    pol.set_policy(UserPolicy("u", ttl_ms=10_000, min_trust=4, max_p95_ms=999999, require_evidence=True, sandbox_caps=()))
    perf = PerfRegistry()
    enf = Enforcement(pol, ca, perf)

    cid = ca.put(b'{}', source="low", trust=2)  # נמוך מדי
    claims = [{"evidence_cid": cid}]
    perf.track("pipeline.total", 12.0)
    try:
        enf.require_response_ok("u", claims, "pipeline.total")
        assert False, "should have raised"
    except EvidenceError:
        pass