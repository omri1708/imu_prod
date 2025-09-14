from fastapi.testclient import TestClient
import services.api.app as apiapp
c = TestClient(apiapp.app)

def test_healthz():
    r = c.get('/healthz'); assert r.status_code == 200 and r.json().get('ok') is True

def test_behavior_cases():
    r = c.post('/compute/calculateTotalSpent', json={"userId": "u1"})
    assert r.status_code == 200
    val = r.json()['score']
    assert abs(val - 150.0) < max(1.0, 0.05*abs(150.0))
    r = c.post('/compute/calculateTotalSpent', json={"userId": "u2"})
    assert r.status_code == 200
    val = r.json()['score']
    assert abs(val - 75.5) < max(1.0, 0.05*abs(75.5))