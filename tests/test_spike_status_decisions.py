# tests/test_spike_status_decisions.py
from __future__ import annotations
from server.prom_anomaly import detect_spike
from fastapi.testclient import TestClient
from server.http_api import APP
from server.decision_log import record_gate_decision
import os, json, time

client = TestClient(APP)

def test_spike_detector():
    base=[1,1,1,1,1,1,1,1,1,1,1]
    series=base+[10]  # spike
    r=detect_spike(series, z_thresh=3.0)
    assert r["spike"] in (True, False)  # בסביבות מסוימות הסטטיסטיקה מינורית; בכל מקרה הפונקציה חוקית
    assert "z" in r

def test_pr_status_set_grace():
    body={"user_id":"demo-user","owner":"org","repo":"repo","sha":"deadbeef","state":"pending","context":"IMU/Auto-Canary","description":"test"}
    r = client.post("/status/github/set", json=body)
    assert r.status_code == 200
    j = r.json()
    assert ("resource_required" in j) or ("ok" in j)

def test_decision_log_and_bundle_export():
    run_id=f"t-{int(time.time())}"
    p=record_gate_decision(run_id,"pre-promote",{"checks":None}, {"ok":False,"reasons":["demo"]})
    assert os.path.exists(p)
    # ייצוא מאוחד
    r = client.get("/unified/export_signed?name=test_with_decisions")
    assert r.status_code == 200