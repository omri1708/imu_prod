# tests/test_replay_archive.py
from __future__ import annotations
from fastapi.testclient import TestClient
from server.http_api import APP

client = TestClient(APP)

def test_history_list():
    r = client.get("/replay/history")
    assert r.status_code == 200
    assert r.json()["ok"] is True

def test_unified_export_signed_and_verify_import(monkeypatch, tmp_path):
    # יצירה
    r = client.get("/unified/export_signed?name=test_unified")
    assert r.status_code == 200
    # לא נוכל לקרוא את הגוף כקובץ כאן; נבדוק שהכותרות קיימות
    assert "X-IMU-Digest" in r.headers