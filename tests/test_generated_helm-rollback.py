# tests/test_generated_helm-rollback.py

from fastapi.testclient import TestClient
from server.http_api import APP
c=TestClient(APP)

def test_helm_rollback_dryrun():
    p={"release":"umbrella","revision":1,"namespace":"prod"}
    r=c.post("/adapters/dry_run", json={"user_id":"demo-user","kind":"helm.rollback","params":p})
    assert r.status_code==200 and r.json()["ok"]