# tests/test_auto_canary_gatekeeper.py
from __future__ import annotations
from fastapi.testclient import TestClient
from server.http_api import APP

client = TestClient(APP)

def test_auto_canary_start_with_gatekeeper_grace():
    # התחלה עם gatekeeper_required – ללא checks אמיתיים (gate ריק) → עובר (gate לא בודק כלום)
    body={"user_id":"demo-user","namespace":"default","app":"imu-app","image":"nginx:alpine","total_replicas":5,"canary_percent":10,
          "probe_url":"http://127.0.0.1:1/","gatekeeper_required":True,"gate":{"evidences":[],"checks":None,"p95":None}}
    r=client.post("/auto_canary/start", json=body)
    assert r.status_code==200 and r.json().get("ok") is True