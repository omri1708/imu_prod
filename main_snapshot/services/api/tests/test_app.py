# -*- coding: utf-8 -*-
from fastapi.testclient import TestClient
import services.api.app as appmod

c = TestClient(appmod.app)

def test_healthz():
    r = c.get("/healthz")
    assert r.status_code == 200 and r.json().get("ok") is True

def test_root_lists_entities():
    r = c.get("/")
    assert r.status_code == 200
    js = r.json()
    assert "entities" in js and isinstance(js["entities"], list)
