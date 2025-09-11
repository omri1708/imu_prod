# tests/test_helm_templates_profiles_and_cert.py
from fastapi.testclient import TestClient
from server.http_api import APP
client = TestClient(APP)

def test_helm_create_with_ingclass_cert_np_strict():
    spec={
      "name":"secure-web",
      "namespace":"default",
      "release":"secure",
      "serviceType":"ClusterIP",
      "replicas":2,
      "containerPort":80,
      "image": {"repository":"nginx","tag":"alpine","pullPolicy":"IfNotPresent"},
      "ingress": {"enabled": True, "className":"nginx", "host":"secure.local", "path":"/", "tlsSecret":"secure-tls"},
      "ingressClass": {"enabled": True, "name":"nginx", "controller":"k8s.io/ingress-nginx"},
      "serviceMonitor": {"enabled": True, "scrapePort":80, "interval":"30s", "path":"/metrics", "scheme":"http", "labels":{"release":"prom-operator"}},
      "networkPolicy": {"enabled": True, "profile":"strict", "allowSameNamespace": True, "ingressCidrs":["10.0.0.0/8"], "egressCidrs":["0.0.0.0/0"]},
      "certManager": {"enabled": True, "issuerKind":"Issuer", "issuerName":"selfsigned", "issuerNamespace":"default", "certificateSecretName":"secure-tls", "dnsNames":["secure.local"]}
    }
    r=client.post("/helm/synth/create", json=spec)
    assert r.status_code==200
    slug=r.json()["slug"]
    g=client.get("/helm/synth/get", params={"slug":slug}).json()
    files=g["files"]
    assert "templates/ingressclass.yaml" in files and "templates/issuer.yaml" in files and "templates/certificate.yaml" in files and "templates/networkpolicy.yaml" in files
    # render
    d=client.post("/helm/synth/dry_template", json={"slug":slug,"name":"secure"}).json()
    assert d["ok"] is True