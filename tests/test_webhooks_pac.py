# tests/test_webhooks_pac.py
from __future__ import annotations
from fastapi.testclient import TestClient
from server.http_api import APP
import json, os

client = TestClient(APP)

def test_github_webhook_push_policy_change(monkeypatch, tmp_path):
    # Disable secret for test env
    if "GITHUB_WEBHOOK_SECRET" in os.environ: del os.environ["GITHUB_WEBHOOK_SECRET"]
    payload={
      "ref":"refs/heads/main",
      "commits":[{"added":["security/policy_rules.yaml"],"modified":[]}]
    }
    r = client.post("/webhooks/github", data=json.dumps(payload), headers={"X-GitHub-Event":"push"})
    assert r.status_code == 200

def test_gitlab_webhook_push_policy_change(monkeypatch):
    if "GITLAB_WEBHOOK_TOKEN" in os.environ: del os.environ["GITLAB_WEBHOOK_TOKEN"]
    payload={
      "object_kind":"push",
      "commits":[{"added":["security/policy_rules.yaml"],"modified":[]}]
    }
    r = client.post("/webhooks/gitlab", data=json.dumps(payload), headers={"X-Gitlab-Event":"Push Hook"})
    assert r.status_code == 200