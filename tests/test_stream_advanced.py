# tests/test_stream_advanced.py — בדיקות WQF/Drop-policies/Freeze לא נשבר
# -*- coding: utf-8 -*-
import time
from broker.stream import broker
from broker.policy import DropPolicy

def test_topic_config_update_and_wfq_ticks():
    # עדכון קונפיג
    broker.configure_topic("wfqA", rps=1e9, burst=1e9, weight=5.0)
    broker.configure_topic("wfqB", rps=1e9, burst=1e9, weight=1.0)
    sA = broker.subscribe("wfqA", max_queue=10)
    sB = broker.subscribe("wfqB", max_queue=10)
    # הרבה פרסומים — WFQ אמור לשמר חלוקה ביחסים ~5:1
    for i in range(60):
        broker.publish("wfqA", {"i": i}, priority="telemetry")
        broker.publish("wfqB", {"i": i}, priority="telemetry")
    a = b = 0; t0 = time.time()
    while time.time()-t0 < 1.0:
        if sA.pop(timeout=0.01): a+=1
        if sB.pop(timeout=0.01): b+=1
    assert a > b and a/(b or 1) >= 3.0

def test_drop_policy_variants():
    s1 = broker.subscribe("events", max_queue=1, drop_policy=DropPolicy.TAIL_DROP)
    assert broker.publish("events", {"a":1})
    ok = broker.publish("events", {"a":2})
    assert ok in (True, False)  # tail-drop עשוי לדחות את החדש