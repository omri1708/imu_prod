# tests/test_p95_wfq_ws.py
# בודק p95 Gate + WFQ submit/poll (ללא WS אמיתי כדי לא להכביד על CI).
from runtime.p95 import GATES
from server.stream_wfq import BROKER

def test_p95_ensure_ok():
    key="adapters.run:unity.build"
    for ms in [10,20,15,50,35,25,30,40,45,60]:
        GATES.observe(key, ms)
    GATES.ensure(key, 500)  # לא אמור לזרוק

def test_wfq_submit_poll():
    BROKER.ensure_topic("timeline", rate=100, burst=50, weight=2)
    ok=0
    for i in range(30):
        if BROKER.submit("timeline","test-producer",{"i":i,"type":"event","note":"t"}, priority=2):
            ok+=1
    batch=BROKER.poll("timeline", max_items=100)
    assert len(batch) == ok or len(batch) == min(ok,100)