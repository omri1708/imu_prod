# tests/test_helm_templates_advanced.py
from fastapi.testclient import TestClient
from server.http_api import APP
client = TestClient(APP)

def test_helm_create_with_ingress_sm_np_and_render():
    spec={
      "name":"adv-web",
      "namespace":"default",
      "release":"adv",
      "serviceType":"ClusterIP",
      "replicas":2,
      "containerPort":80,
      "image": {"repository":"nginx","tag":"alpine","pullPolicy":"IfNotPresent"},
      "ingress": {"enabled": True, "className":"nginx", "host":"adv.local", "path":"/", "tlsSecret":"adv-tls"},
      "serviceMonitor": {"enabled": True, "scrapePort":80, "interval":"30s", "path":"/metrics", "scheme":"http", "labels": {"team":"platform"}},
      "networkPolicy": {"enabled": True, "allowSameNamespace": True, "ingressCidrs":["10.0.0.0/8"], "egressCidrs":["0.0.0.0/0"]},
      "hpa": True, "hpaMin": 2, "hpaMax": 5, "hpaCpu": 75
    }
    r=client.post("/helm/synth/create", json=spec)
    assert r.status_code==200
    slug=r.json()["slug"]
    g=client.get("/helm/synth/get", params={"slug":slug}).json()
    files=g["files"]; assert "templates/ingress.yaml" in files and "templates/servicemonitor.yaml" in files and "templates/networkpolicy.yaml" in files
    d=client.post("/helm/synth/dry_template", json={"slug":slug,"name":"adv"}).json()
    assert d["ok"] is True