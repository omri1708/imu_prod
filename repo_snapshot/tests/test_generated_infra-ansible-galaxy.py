
# tests/test_generated_infra-ansible-galaxy.py
from fastapi.testclient import TestClient
from server.http_api import APP
client = TestClient(APP)

def test_ansible_galaxy_dryrun():
    params={"requirements":"./requirements.yml","dest":"./roles","server_opt":"","token_opt":"","extra_opt":""}
    r=client.post("/adapters/dry_run", json={"user_id":"demo-user","kind":"infra.ansible.galaxy","params":params})
    assert r.status_code==200
    j=r.json(); assert j["ok"] and "ansible-galaxy install -r ./requirements.yml -p ./roles" in j["cmd"]



