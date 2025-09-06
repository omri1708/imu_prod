# tests/test_scheduler_tasks.py
from __future__ import annotations
from fastapi.testclient import TestClient
from server.http_api import APP
import time, json

client = TestClient(APP)

def test_schedule_unified_export_and_attest_grace():
    # start scheduler
    client.post("/scheduler/start")
    # schedule unified export in +0.3s
    r = client.post("/scheduler/create", json={
        "user_id":"demo-user","kind":"unified.export_signed","mode":"at","at_ts": time.time()+0.3,
        "args":{"name": f"test_export_{int(time.time())}"}
    })
    assert r.status_code == 200
    # schedule supplychain attest (will return resource_required without cosign; still should run)
    r2 = client.post("/scheduler/create", json={
        "user_id":"demo-user","kind":"supplychain.attest","mode":"at","at_ts": time.time()+0.4,
        "args":{"image":"nginx:alpine","predicate_path":"sbom/cyclonedx_demo.json"}
    })
    assert r2.status_code == 200
    time.sleep(1.0)
    # state endpoint alive
    st = client.get("/scheduler/state").json()
    assert st["ok"] is True