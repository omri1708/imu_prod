# tests/test_pack_a_end2end.py
# בודק שה־demos שולחים dry-run מוצלח (ללא הרצה אמיתית), ושה-API עומד בחוזה.
from __future__ import annotations
import subprocess, sys, os, json
import pytest
from fastapi.testclient import TestClient
from server.http_api import APP

client = TestClient(APP)

@pytest.mark.parametrize("path,body", [
    ("/adapters/dry_run", {"user_id":"demo-user","kind":"unity.build","params":{"project":"/proj","target":"Android","method":"Builder.PerformBuild","version":"2022.3.44f1","log":"/tmp/u.log"}}),
    ("/adapters/dry_run", {"user_id":"demo-user","kind":"android.gradle","params":{"flavor":"Release","buildType":"Aab","keystore":"/p/k.jks"}}),
    ("/adapters/dry_run", {"user_id":"demo-user","kind":"ios.xcode","params":{"workspace":"App.xcworkspace","scheme":"App","config":"Release"}}),
    ("/adapters/dry_run", {"user_id":"demo-user","kind":"k8s.kubectl.apply","params":{"manifest":"./deploy.yaml","namespace":"prod"}}),
    ("/adapters/dry_run", {"user_id":"demo-user","kind":"cuda.nvcc","params":{"src":"kern.cu","out":"kern"}}),
])
def test_api_dry_contracts(path, body):
    r = client.post(path, json=body)
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] and "cmd" in j and "evidence" in j

def test_cap_request_known_caps():
    for cap in ("unity.hub","jdk","gradle","nodejs","kubectl","docker","cuda"):
        r = client.post("/capabilities/request", json={"user_id":"demo-user","capability":cap})
        assert r.status_code == 200
        assert r.json()["ok"] is True