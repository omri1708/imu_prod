from fastapi.testclient import TestClient
import services.api.app as apiapp
c = TestClient(apiapp.app)

def test_healthz():
    r = c.get('/healthz'); assert r.status_code == 200 and r.json().get('ok') is True

def test_behavior_cases():
    r = c.post('/compute/CalculateTotalPrice', json={"orderId": 1})
    assert r.status_code == 200
    val = r.json()['score']
    assert abs(val - 100.0) < max(1.0, 0.05*abs(100.0))
    r = c.post('/compute/CalculateTotalPrice', json={"orderId": 2})
    assert r.status_code == 200
    val = r.json()['score']
    assert abs(val - 50.0) < max(1.0, 0.05*abs(50.0))