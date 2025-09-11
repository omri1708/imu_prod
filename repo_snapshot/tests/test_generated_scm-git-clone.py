# tests/test_generated_scm-git-clone.py
from fastapi.testclient import TestClient
from server.http_api import APP
client = TestClient(APP)

def test_git_clone_dryrun():
    params={"repo":"https://github.com/org/repo.git","dest":"./repo","branch_opt":" -b main","depth_opt":" --depth 1"}
    r=client.post("/adapters/dry_run", json={"user_id":"demo-user","kind":"scm.git.clone","params":params})
    assert r.status_code==200
    j=r.json(); assert j["ok"] and "git clone https://github.com/org/repo.git ./repo -b main --depth 1".replace("  "," ") in j["cmd"].replace("  "," ")