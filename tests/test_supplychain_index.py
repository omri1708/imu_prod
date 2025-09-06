# tests/test_supplychain_index.py
from __future__ import annotations
from fastapi.testclient import TestClient
from server.http_api import APP

client = TestClient(APP)

def test_index_put_get():
    r = client.post("/supplychain/index/put", json={"image":"busybox","digest":"sha256:deadbeef","envelope_path":".imu/provenance/env_deadbeef.json"})
    assert r.json()["ok"] is True
    g = client.get("/supplychain/index/get", params={"image":"busybox"}).json()
    assert g["ok"] and g["items"]