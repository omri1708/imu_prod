# tests/test_runtime_p95_wfq.py
from runtime.p95 import GATES
from server.stream_wfq import WFQBroker

def test_p95_gate():
    key="respond"
    for ms in [50,60,55,52,51,100,120,90,80,70,65,62,61]:
        GATES.observe(key, ms)
    # תקרה 200ms—לא יכשל
    GATES.ensure(key, 200)

def test_wfq():
    b=WFQBroker(global_rate=100, global_burst=20)
    b.ensure_topic("timeline", rate=50, burst=10, weight=2)
    b.ensure_topic("logs", rate=10, burst=5, weight=1)
    # הזרמה
    ok_t=ok_l=0
    for i in range(30):
        ok_t += 1 if b.submit("timeline", "prodA", {"i":i}, priority=1) else 0
        ok_l += 1 if b.submit("logs", "prodB", {"i":i}, priority=5) else 0
    # timeline אמור לקבל יותר פריטים מאשר logs
    tl=len(b.poll("timeline", 100))
    lg=len(b.poll("logs", 100))
    assert tl >= lg