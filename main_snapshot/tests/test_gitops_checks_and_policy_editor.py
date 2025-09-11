# tests/test_gitops_checks_and_policy_editor.py
from __future__ import annotations
from fastapi.testclient import TestClient
from server.http_api import APP

client = TestClient(APP)

def test_policy_editor_roundtrip(tmp_path):
    y = """
user_policies:
  demo-user:
    default_net: deny
    default_fs:  deny
    net_allow:
      - {host: "127.0.0.1", ports: [8000,8765]}
    fs_allow:
      - {path: "./", mode: "rw", ttl_seconds: 3600}
"""
    r = client.post("/policy/yaml", json={"yaml_text": y})
    assert r.status_code == 200
    g = client.get("/policy/yaml")
    assert g.status_code == 200
    assert "demo-user" in g.json()["yaml"]

def test_gh_checks_resource_required_or_ok():
    r = client.post("/gitops/guard/github/checks", json={"owner":"octocat","repo":"hello-world","ref":"HEAD"})
    assert r.status_code == 200
    j = r.json()
    assert ("resource_required" in j) or ("checks" in j)