# tests/test_key_admin_and_archive.py
from __future__ import annotations
from fastapi.testclient import TestClient
from server.http_api import APP

client = TestClient(APP)

def test_keys_list_rotate_activate():
    r = client.get("/keys/")
    assert r.status_code == 200
    before = r.json()["keys"]
    r2 = client.post("/keys/rotate", json={"comment":"pytest"})
    assert r2.json()["ok"] is True
    kid = r2.json()["active"]
    r3 = client.post("/keys/activate", json={"kid": kid})
    assert r3.json()["ok"] is True

def test_public_bundle_and_import():
    b = client.get("/keys/public_bundle").json()
    assert b["ok"] is True
    # import the first public back under a new kid, for test
    kid = list(b["bundle"].keys())[0]
    pem = b["bundle"][kid]
    r = client.post("/keys/import_public", json={"kid": kid+"x", "pub_pem": pem})
    assert r.json()["ok"] is True

def test_archive_export():
    r = client.get("/archive/export")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"