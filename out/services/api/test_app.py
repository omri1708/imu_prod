from fastapi.testclient import TestClient
import services.api.app as apiapp
c = TestClient(apiapp.app)

def test_healthz():
    r = c.get('/healthz'); assert r.status_code == 200 and r.json().get('ok') is True

def test_crud_entity():
    r = c.post('/users', json={'id': 1})
    assert r.status_code in (200, 409)
    r = c.get('/users'); assert r.status_code == 200

def test_behavior_cases():
    r = c.post('/compute/CalculateTotalPrice', json={"productId": "p1", "quantity": 3})
    assert r.status_code == 200
    val = r.json()['score']
    assert abs(val - 45.0) < max(1.0, 0.05*abs(45.0))
    r = c.post('/compute/CalculateTotalPrice', json={"productId": "p2", "quantity": 2})
    assert r.status_code == 200
    val = r.json()['score']
    assert abs(val - 40.0) < max(1.0, 0.05*abs(40.0))