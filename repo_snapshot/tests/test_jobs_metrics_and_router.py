# tests/test_jobs_metrics_and_router.py
from __future__ import annotations
from fastapi.testclient import TestClient
from server.http_api import APP
from server.stream_policy_router import _adjust_once
from server.stream_wfq import BROKER

client = TestClient(APP)

def test_jobs_summary_endpoint():
    r=client.get("/metrics/jobs/summary")
    assert r.status_code==200
    j=r.json()
    assert j["ok"] is True and "kinds" in j

def test_policy_router_adjust_once_runs():
    # מכניסים קצת אירועים לנושא timeline ו-logs ואז adjust
    BROKER.ensure_topic("timeline", rate=50, burst=100, weight=2)
    BROKER.ensure_topic("logs", rate=50, burst=100, weight=1)
    for i in range(600):
        BROKER.submit("timeline","test",{"type":"event","ts":0,"note":"x"}, priority=5)
    adj=_adjust_once()
    assert "timeline" in adj