# tests/test_generated_infra-ansible-apply.py
from fastapi.testclient import TestClient
from server.http_api import APP
client = TestClient(APP)

def test_ansible_apply_dryrun():
    params={"inventory":"./hosts.ini","playbook":"site.yml","tags_opt":" --tags web","extra_opt":""}
    r=client.post("/adapters/dry_run", json={"user_id":"demo-user","kind":"infra.ansible.apply","params":params})
    assert r.status_code==200
    j=r.json(); assert j["ok"] and "ansible-playbook -i ./hosts.ini site.yml --tags web" in j["cmd"]