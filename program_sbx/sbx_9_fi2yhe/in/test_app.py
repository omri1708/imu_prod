from fastapi.testclient import TestClient
import app
c = TestClient(app.app)

def test_smoke_no_behavior():
    assert True