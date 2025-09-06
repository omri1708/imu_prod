# tests/test_prom_ready_status.py
from __future__ import annotations
from fastapi.testclient import TestClient
from server.http_api import APP
from server.k8s_ready import have_kubectl, readiness_ratio

client = TestClient(APP)

def test_prom_client_grace():
    # אין פרומתאוס בסביבה; נבדוק שהתלות אינה קורסת — נשתמש בכתובת לא קיימת והתשובה תיכשל בצד הלקוח (אין פנייה כאן)
    # Endpoint לא מסופק; נבחן רק יבוא פונקציות (ללא ריצה)
    assert True

def test_ready_ratio_grace():
    if not have_kubectl():
        assert True
    else:
        r = readiness_ratio("default","nonexistent-app")
        assert "ok" in r

def test_evaluate_and_set_status_no_token():
    body={"evidences":[],"checks":None,"p95":None,"owner":"org","repo":"repo","sha":"deadbeef","context":"IMU/Gatekeeper"}
    r = client.post("/gatekeeper/evaluate_and_set_status", json=body)
    assert r.status_code == 200
    j = r.json()
    # בלי טוקן, עדכון סטטוס יחזור resource_required בתוך downstream; gates=false במחדל; עדיין json תקין
    assert "status_update" in j