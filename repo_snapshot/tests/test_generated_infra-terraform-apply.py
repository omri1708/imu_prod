# tests/test_generated_infra-terraform-apply.py
from fastapi.testclient import TestClient
from server.http_api import APP
client = TestClient(APP)

def test_tf_apply_dryrun():
    params={"dir":"./infra","varfile_opt":" -var-file=prod.tfvars","var_opt":"","backend_opt":"","approve_opt":" -auto-approve"}
    r=client.post("/adapters/dry_run", json={"user_id":"demo-user","kind":"infra.terraform.apply","params":params})
    assert r.status_code==200
    j=r.json(); assert j["ok"] and "terraform -chdir=./infra apply" in j["cmd"]