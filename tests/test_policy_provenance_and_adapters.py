# tests/test_policy_provenance_and_adapters.py
# -*- coding: utf-8 -*-
import os, json
from fastapi.testclient import TestClient
from server.http_api import app
from policy.policy_engine import policy_store

client = TestClient(app)

def test_policy_put_and_enforce():
    r = client.post("/policy/put", json={
        "user_id":"u1","trust":"high","ttl_seconds":3600,"p95_ms_max":500,
        "net_allow":["^api\\.example\\.com$"],"fs_allow":[r"^\.?data/"]
    })
    assert r.json()["ok"] is True
    u = policy_store.get("u1")
    assert u.trust=="high" and u.limits.ttl_seconds==3600

def test_provenance_ingest_and_sign():
    os.makedirs("data",exist_ok=True)
    path="data/sample.txt"; open(path,"w").write("hello")
    client.post("/provenance/keygen/test")
    r = client.post("/provenance/ingest", json={"path":path,"kind":"file","trust":"high","signer":"test"})
    j = r.json()
    assert "hash" in j and "evidence" in j

def test_capabilities_dry_runs():
    r = client.post("/capabilities/request", json={"capability":"k8s-dry-run","args":{}})
    assert "ok" in r.json()
    # Android dry-run will fail on purpose without gradle/jdk, but returns actionable why list
    r = client.post("/capabilities/request", json={"capability":"android-dry-run","args":{"project_dir":"."}})
    assert "why" in r.json()