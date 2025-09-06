# tests/test_generated_helm_templates.py
from fastapi.testclient import TestClient
from server.http_api import APP
client=TestClient(APP)

def test_helm_template_and_upgrade_dry():
    params_t={"name":"myrel","chart_dir":"./helm/generated/hello-web","values_file":"./helm/generated/hello-web/values.yaml"}
    r=client.post("/adapters/dry_run", json={"user_id":"demo-user","kind":"helm.template","params":params_t})
    assert r.status_code==200
    j=r.json(); assert j["ok"] and "helm template myrel" in j["cmd"]
    params_u={"release":"myrel","chart_dir":"./helm/generated/hello-web","namespace":"default","values_file":"./helm/generated/hello-web/values.yaml","extra_opt":""}
    r2=client.post("/adapters/dry_run", json={"user_id":"demo-user","kind":"helm.upgrade","params":params_u})
    assert r2.status_code==200 and r2.json()["ok"]