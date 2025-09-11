# tests/test_attest_verify_routes.py
from __future__ import annotations
from fastapi.testclient import TestClient
from server.http_api import APP

client = TestClient(APP)

def test_attest_returns_ok_or_resource_required():
    # בסביבת CI ללא cosign נקבל resource_required; אחרת ok
    r = client.post("/supplychain/index/attest", json={"image":"nginx:alpine","predicate_path":"sbom/cyclonedx_demo.json"})
    assert r.status_code == 200
    j = r.json()
    assert ("resource_required" in j) or ("ok" in j)

def test_verify_keyless_route():
    r = client.post("/supplychain/verify/docker/keyless", json={"image":"nginx:alpine"})
    assert r.status_code == 200
    j = r.json()
    assert ("resource_required" in j) or ("ok" in j)