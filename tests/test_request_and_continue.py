# tests/test_request_and_continue.py
from fastapi.testclient import TestClient
from server.main import app

client = TestClient(app)

def test_capabilities_list():
    r = client.get("/capabilities/list")
    assert r.status_code == 200
    caps = r.json()["capabilities"]
    for c in ["android","ios","unity","cuda","k8s"]:
        assert c in caps