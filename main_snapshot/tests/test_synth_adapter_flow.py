# tests/test_synth_adapter_flow.py
from __future__ import annotations
from fastapi.testclient import TestClient
from server.http_api import APP

client = TestClient(APP)

SPEC = {
  "name": "DB Migrator",
  "kind": "db.migrate",
  "version": "1.0.0",
  "description": "Run database migrations via migrate CLI",
  "params": {
    "db_url":    {"type":"string", "required":True},
    "dir":       {"type":"string", "required":True},
    "timeout_s": {"type":"number", "default":60}
  },
  "os_templates": {
    "any": "migrate -database {db_url} -path {dir} up"
  },
  "examples": {"db_url": "postgres://user:pass@localhost/db?sslmode=disable", "dir":"./migrations", "timeout_s":60},
  "capabilities": ["migrate"]
}

def test_synth_create_and_dryrun():
    r = client.post("/synth/adapter/create", json=SPEC)
    assert r.status_code == 200
    r2 = client.get("/synth/adapter/list")
    assert r2.status_code == 200 and "db.migrate" in r2.json().get("kinds",[])
    # dry-run
    r3 = client.post("/adapters/dry_run", json={"user_id":"demo-user","kind":"db.migrate","params":SPEC["examples"]})
    assert r3.status_code == 200
    j = r3.json()
    assert j["ok"] and "migrate -database" in j["cmd"]