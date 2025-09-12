from fastapi.testclient import TestClient
import services.api.app as apiapp
c = TestClient(apiapp.app)

def test_healthz():
    r = c.get('/healthz'); assert r.status_code == 200 and r.json().get('ok') is True

def test_behavior_cases():
    r = c.post('/compute/CalculateTotalPrice', json={"productId": 2})
    assert r.status_code == 200
    val = r.json()['score']
    assert abs(val - 50.0) < max(1.0, 0.05*abs(50.0))
    r = c.post('/compute/CalculateTotalPrice', json={"productId": 3})
    assert r.status_code == 200
    val = r.json()['score']
    assert abs(val - 75.0) < max(1.0, 0.05*abs(75.0))