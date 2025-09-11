# tests/test_hotload_and_runbook.py
from __future__ import annotations
import os, time, json, tempfile
from fastapi.testclient import TestClient
from server.http_api import APP
from policy.policy_hotload import _apply_cfg

client = TestClient(APP)

def test_hotload_apply_cfg():
    cfg = {
        "user_policies": {
            "u1": {
                "default_net": "deny",
                "default_fs": "deny",
                "net_allow": [{"host":"127.0.0.1","ports":[8000]}],
                "fs_allow": [{"path":"./tmp","mode":"rw","ttl_seconds":10}]
            }
        }
    }
    _apply_cfg(cfg)  # מחיל אל POLICY_DB/FS_DB
    # ודאו שה־policy מוצג
    r = client.get("/api/policy/network/u1")
    assert r.status_code == 200
    j = r.json()
    assert j["default_deny"] is True
    assert j["rules"][0]["host"] == "127.0.0.1"

def test_runbook_unity_k8s_dry():
    # לא תלויות בכלים: runbook יתן dry-run (exec=false עקיף דרך /adapters/run)
    body={"user_id":"demo-user","project_dir":"/proj","target":"Android","namespace":"default","name":"unity-app"}
    r = client.post("/runbook/unity_k8s", json=body)
    assert r.status_code == 200
    j = r.json()
    assert "unity" in j and "k8s" in j