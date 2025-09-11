# tests/test_canary_and_merge_guard.py
from __future__ import annotations
from fastapi.testclient import TestClient
from server.http_api import APP
import shutil

client = TestClient(APP)

def test_canary_deploy_grace():
    j = client.post("/canary/deploy", json={"user_id":"demo-user","namespace":"default","app":"x","image":"nginx:alpine","total_replicas":5,"canary_percent":20,"dry":True}).json()
    assert j.get("ok") is True

def test_merge_guard_requires_gates():
    body={"user_id":"demo-user",
          "evidences":[{"digest":"deadbeef"*8,"min_trust":0.9}],
          "checks":{"owner":"org","repo":"repo","ref":"HEAD","required":["build"],"mode":"all","token_env":"IMU_NO_TOKEN"},
          "p95":{"keys":["adapters.run:unity.build"],"ceiling_ms":5000},
          "owner":"org","repo":"repo","pr_number":1}
    r = client.post("/merge_guard/github", json=body)
    # אין טוקן → gates יחזירו false, קוד 412
    assert r.status_code in (412, 400)