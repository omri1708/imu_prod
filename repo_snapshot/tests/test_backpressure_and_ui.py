# tests/test_backpressure_and_ui.py — בדיקות Back-pressure/Throttling/עדיפויות
# -*- coding: utf-8 -*-
import time
from broker.stream import broker

def test_backpressure_global_guard():
    # הורדנו קצב כדי להכריח שסתום
    broker.global_bucket = broker.global_bucket.__class__(rps=1.0, burst=2)
    accepted = 0; rejected = 0
    for i in range(10):
        ok = broker.publish("logs", {"i": i}, priority="logs")
        if ok: accepted += 1
        else: rejected += 1
    # נוודא שיש לפחות דחייה אחת
    assert rejected >= 1
    assert accepted >= 1

def test_priority_drop_on_full_queue():
    sub = broker.subscribe("events", max_queue=5)
    # מלא את התור בלוגים (עדיפות נמוכה)
    for i in range(5):
        broker.publish("events", {"i": i, "kind":"low"}, priority="logs")
    # עכשיו פרסם בעדיפות גבוהה — אמור להיכנס ולהחליף נמוך
    broker.publish("events", {"i": 999, "kind":"high"}, priority="logic")
    # קרא את כל ההודעות — ודא שהגבוהה קיימת
    got = []
    t0 = time.time()
    while time.time()-t0 < 1.0:
        ev = sub.pop(timeout=0.1)
        if ev: got.append(ev)
        if len(got) >= 6: break
    assert any(e["kind"] == "high" for e in got)