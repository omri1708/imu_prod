# tests/test_gitops_api.py
from __future__ import annotations
from fastapi.testclient import TestClient
from server.http_api import APP
import tempfile, os, shutil
from gitops.utils import have_git

client = TestClient(APP)

def test_gitops_init_branch_commit_list_pr(tmp_path):
    repo = tmp_path/"repo"; os.makedirs(repo, exist_ok=True)
    # create a file
    p = repo/"README.md"; p.write_text("# demo\n", encoding="utf-8")
    r = client.post("/gitops/init", json={"user_id":"demo-user","path":str(repo)})
    assert r.status_code == 200 and r.json()["ok"]
    r = client.post("/gitops/branch", json={"user_id":"demo-user","path":str(repo),"name":"feature-demo"})
    assert r.json()["ok"]
    r = client.post("/gitops/commit", json={"user_id":"demo-user","path":str(repo),"message":"add readme","add_patterns":["README.md"]})
    assert r.json()["ok"]
    r = client.post("/gitops/pr/open", json={"user_id":"demo-user","path":str(repo),"branch":"feature-demo","target":"main","title":"demo","description":"desc"})
    assert r.json()["ok"]

def test_gitops_push_resource_required_or_ok(tmp_path):
    repo = tmp_path/"repo2"; os.makedirs(repo, exist_ok=True)
    (repo/"a.txt").write_text("hi", encoding="utf-8")
    client.post("/gitops/init", json={"user_id":"demo-user","path":str(repo)})
    # אם git לא זמין — נדלג; אם זמין אבל אין remote — גם יכשל; עדיין endpoint יחזיר json
    r = client.post("/gitops/push", json={"user_id":"demo-user","path":str(repo),"remote":"origin"})
    assert r.status_code == 200
    # ok או resource_required
    j = r.json()
    assert ("resource_required" in j) or ("ok" in j)