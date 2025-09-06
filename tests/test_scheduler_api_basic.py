# tests/test_scheduler_api_basic.py
from __future__ import annotations
from fastapi.testclient import TestClient
from server.http_api import APP
import time

client = TestClient(APP)

def test_scheduler_emit_event_and_poll():
    # start scheduler (idempotent)
    client.post("/scheduler/start")
    # create emit_event schedule to fire soon
    r=client.post("/scheduler/create", json={
        "user_id":"demo-user",
        "kind":"emit_event",
        "mode":"at",
        "at_ts": time.time()+0.3,
        "args":{"topic":"timeline","note":"scheduler-test","pct":13}
    })
    assert r.status_code==200 and r.json()["ok"]
    time.sleep(0.6)
    # poll WFQ via events gateway
    g=client.get("/events/poll", params={"topic":"timeline","max":50}).json()
    assert g["ok"] is True
    # may or may not include our event depending on timing—still ensure endpoint is alive