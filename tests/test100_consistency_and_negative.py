# imu_repo/tests/test100_consistency_and_negative.py
from __future__ import annotations
import os, shutil, json
from grounded.claims import current
from ui.dsl import Page, Component
from engine.provenance_gate import enforce_evidence_gate, GateFailure
from engine.rollout_guard import run_negative_suite, RolloutBlocked

def setup_module(module=None):
    try: current().clear()
    except Exception: pass

def _simple_page():
    # טבלה שקושרה ל־endpoint, וטקסט סטטי (לבדיקה)
    return Page(
        title="Consistency100",
        components=[
            Component(kind="table", id="orders", props={"endpoint":"https://api.example.com/orders"}),
            Component(kind="text", id="t", props={"text":"Hello"})
        ]
    )

def test_block_without_evidence():
    page = _simple_page()
    # אין ראיות בכלל → אמור להיחסם
    try:
        run_negative_suite(page, [], policy={"min_trust":0.75,"min_sources":2,"max_ttl_s":86400})
        assert False, "expected RolloutBlocked"
    except RolloutBlocked:
        pass

def test_pass_with_fresh_multi_sources_and_trust():
    page = _simple_page()
    # אפס ראיות קודמות
    try: current().clear()
    except Exception: pass
    # שלוש ראיות שונות לאותו endpoint (דיוורסיטי + אמון)
    current().add_evidence("srcA", {"kind":"docs", "source_url":"https://api.example.com", "payload":{"spec":"v1"}, "ttl_s":86400, "trust":0.80})
    current().add_evidence("srcB", {"kind":"ui",   "source_url":"https://api.example.com/orders", "payload":{"ok":True}, "ttl_s":86400, "trust":0.78})
    current().add_evidence("srcC", {"kind":"config","source_url":"https://api.example.com/.well-known", "payload":{"auth":"mtls"}, "ttl_s":86400, "trust":0.82})

    evs = current().snapshot()
    res = run_negative_suite(page, evs, policy={"min_trust":0.75,"min_sources":2,"max_ttl_s":86400})
    assert res["ok"] and res["sources"] >= 2 and res["agg_trust"] >= 0.75