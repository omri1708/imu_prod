# tests/test_auto_canary_merge_guard.py
from __future__ import annotations
from fastapi.testclient import TestClient
from server.http_api import APP
import time

client = TestClient(APP)

def test_auto_canary_start_status_grace():
    # פרוב סינתטי ללוקאל (ייתכן 404 → error_rate גבוה → rollback; עדיין אמור להשיב JSON)
    r = client.post("/auto_canary/start", json={"user_id":"demo-user","namespace":"default","app":"x","image":"nginx:alpine","total_replicas":5,"canary_percent":10,"probe_url":"http://127.0.0.1:1/"})
    assert r.status_code == 200
    time.sleep(0.2)
    s = client.get("/auto_canary/status").json()
    assert s["ok"] is True

def test_merge_guard_github_gates_fail_no_token():
    body={"user_id":"demo-user","evidences":[],"checks":{"owner":"org","repo":"repo","ref":"HEAD","required":["build"],"mode":"all","token_env":"IMU_NO_TOKEN"},
          "owner":"org","repo":"repo","pr_number":1}
    r = client.post("/merge_guard/github", json=body)
    assert r.status_code in (412, 400)