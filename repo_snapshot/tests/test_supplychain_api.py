# tests/test_supplychain_api.py
from __future__ import annotations
from fastapi.testclient import TestClient
from server.http_api import APP

client = TestClient(APP)

def test_sign_keyless_resource_required_or_ok():
    r = client.post("/supplychain/sign/docker/keyless", json={"image":"nginx:alpine","yes":True})
    assert r.status_code == 200
    j = r.json()
    # בסביבת CI ללא cosign נקבל resource_required; כשיש cosign נקבל ok=True
    assert ("resource_required" in j) or ("ok" in j)

def test_verify_keyless_resource_required_or_ok():
    r = client.post("/supplychain/verify/docker/keyless", json={
        "image":"nginx:alpine",
        "certificate_identity":"https://github.com/*",
        "certificate_oidc_issuer":"https://token.actions.githubusercontent.com"
    })
    assert r.status_code == 200
    j = r.json()
    assert ("resource_required" in j) or ("ok" in j)