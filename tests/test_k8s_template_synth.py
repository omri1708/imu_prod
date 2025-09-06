# tests/test_k8s_template_synth.py
from fastapi.testclient import TestClient
from server.http_api import APP
client = TestClient(APP)

def test_k8s_template_create_and_dry():
    spec={
      "name":"hello-web",
      "namespace":"default",
      "labels":{"tier":"web"},
      "replicas":2,
      "service_type":"ClusterIP",
      "container":{"name":"web","image":"nginx:alpine","port":80,"env":{"GREETING":"hi"}},
      "hpa":True,"hpa_min":2,"hpa_max":5,"hpa_cpu":75
    }
    r=client.post("/k8s/synth/create", json=spec)
    assert r.status_code==200
    slug=r.json()["slug"]
    g=client.get("/k8s/synth/get", params={"slug":slug}).json()
    assert g["ok"] and "deployment.yaml" in g["files"]
    d=client.post("/k8s/synth/dry_apply", json={"slug":slug,"user_id":"demo-user"}).json()
    assert d["ok"]