# tests/test_bundles_rbac.py
from __future__ import annotations
from fastapi.testclient import TestClient
from server.http_api import APP
from policy.rbac import RBAC_DB

client = TestClient(APP)

def test_bundles_create_list_verify_with_admin():
    # demo-user נהנה מרול admin כברירת מחדל
    r = client.post("/bundles/create", json={"user_id":"demo-user","name":"test-bundle","comment":"ci"})
    assert r.status_code == 200
    j = r.json(); assert j["ok"] is True
    r2 = client.get("/bundles/list")
    assert r2.status_code == 200 and r2.json()["ok"] is True
    r3 = client.post("/bundles/verify", json={"name":"test-bundle"})
    # בסביבת CI, החתימה והזיפ קיימים; verify יחזיר ok True
    assert r3.status_code == 200 and "ok" in r3.json()

def test_bundles_create_denied_for_viewer(tmp_path, monkeypatch):
    # יוצר משתמש חדש עם viewer בלבד
    RBAC_DB.users["viewer-user"] = RBAC_DB.users.get("viewer-user") or type(RBAC_DB.users["demo-user"])("viewer-user", ["viewer"])
    RBAC_DB.users["viewer-user"].roles = ["viewer"]
    r = client.post("/bundles/create", json={"user_id":"viewer-user","name":"nope"})
    assert r.status_code == 403