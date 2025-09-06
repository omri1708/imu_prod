# tests/test_gatekeeper_private_fetch.py
from __future__ import annotations
from server.private_repo_fetch import fetch_github_file, fetch_gitlab_file
from fastapi.testclient import TestClient
from server.http_api import APP

client = TestClient(APP)

def test_private_fetch_without_token():
    # ללא טוקן צפוי resource_required
    g = fetch_github_file("octocat","hello-world","README.md","main", token_env="IMU_NONEXIST")
    assert g["ok"] is False and g.get("resource_required")=="IMU_NONEXIST"
    l = fetch_gitlab_file("123","README.md","main", token_env="IMU_NONEXIST")
    assert l["ok"] is False and l.get("resource_required")=="IMU_NONEXIST"

def test_gatekeeper_denied_without_anything():
    r = client.post("/gatekeeper/evaluate", json={"evidences":[{"digest":"deadbeef"*8,"min_trust":0.9}]})
    assert r.status_code == 200
    assert r.json()["ok"] is False