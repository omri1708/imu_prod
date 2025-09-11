from fastapi.testclient import TestClient
from server.http_api import APP
client = TestClient(APP)

def test_controlplane_dry_endpoint():
    r = client.post("/controlplane/dry", json={"release":"imu","namespace":"default","values_file":"helm/control-plane/values.yaml"})
    assert r.status_code==200
    j = r.json()
    assert "cmd" in j  # מוודא שההתלכדות נעשית דרך adapter

def test_controlplane_deploy_grace():
    r = client.post("/controlplane/deploy", json={"user_id":"demo-user","release":"imu","namespace":"default","values_file":"helm/control-plane/values.yaml","execute":True})
    assert r.status_code==200