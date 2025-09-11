# tests/test_pack_a_dry_end2end.py
# בודק שה־dry-run לכל האדפטרים מחזיר פקודות תקינות + Evidences.
from __future__ import annotations
import pytest
from fastapi.testclient import TestClient
from server.http_api import APP

client = TestClient(APP)

CASES = [
    ("unity.build", {"project":"/proj","target":"Android","method":"Builder.PerformBuild","version":"2022.3.44f1","log":"/tmp/u.log"}),
    ("android.gradle", {"flavor":"Release","buildType":"Aab","keystore":"/proj/k.jks"}),
    ("ios.xcode", {"workspace":"App.xcworkspace","scheme":"App","config":"Release"}),
    ("k8s.kubectl.apply", {"manifest":"./deploy.yaml","namespace":"prod"}),
    ("cuda.nvcc", {"src":"kern.cu","out":"kern"}),
]

@pytest.mark.parametrize("kind,params", CASES)
def test_all_dry(kind, params):
    r = client.post("/adapters/dry_run", json={"user_id":"demo-user","kind":kind,"params":params})
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is True
    assert "cmd" in j
    assert j["cmd"]
    assert len(j["evidence"]) >= 1