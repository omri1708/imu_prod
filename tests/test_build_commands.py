# tests/test_build_commands.py
# בדיקות קונקרטיות (pytest) ל-dry-run ול-capability mappings.

from __future__ import annotations
import pytest
from server.http_api import APP
from fastapi.testclient import TestClient

client = TestClient(APP)

@pytest.mark.parametrize("kind,params", [
    ("unity.build", {"project":"/proj", "target":"Android", "method":"Builder.PerformBuild", "version":"2022.3.44f1", "log":"/tmp/u.log"}),
    ("android.gradle", {"flavor":"Release", "buildType":"Aab", "keystore":"/proj/keys/app.keystore"}),
    ("ios.xcode", {"workspace":"App.xcworkspace", "scheme":"App", "config":"Release"}),
    ("k8s.kubectl.apply", {"manifest":"./k.yaml", "namespace":"prod"}),
    ("cuda.nvcc", {"src":"kern.cu", "out":"kern"}),
])
def test_dry_run_templates(kind, params):
    r = client.post("/adapters/dry_run", json={"user_id":"demo-user","kind":kind,"params":params})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["ok"] is True
    assert "cmd" in j and isinstance(j["cmd"], str)
    assert "evidence" in j and len(j["evidence"]) >= 1
    # אין טוקנים מסוכנים
    assert " rm -rf " not in j["cmd"]

def test_capability_request_maps():
    for cap in ["unity.hub","jdk","gradle","nodejs","kubectl","docker","cuda"]:
        r = client.post("/capabilities/request", json={"user_id":"demo-user","capability":cap})
        assert r.status_code == 200
        j = r.json()
        assert j["ok"] is True
        assert "command" in j
        assert len(j["evidence"]) >= 1