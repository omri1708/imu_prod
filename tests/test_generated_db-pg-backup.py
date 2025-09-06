# tests/test_generated_db-pg-backup.py
from fastapi.testclient import TestClient
from server.http_api import APP

client = TestClient(APP)

def test_db_pg_backup_dryrun():
    params = {
        "db_url": "postgres://user:pass@localhost:5432/app?sslmode=disable",
        "out": "/tmp/app.sql",
        "format": "p",
        # suffixes (assembled as strings; allowed by our contract via additional placeholders):
        "schema_opt": "",
        "owner_opt": " --no-owner",
        "jobs_opt": ""
    }
    r = client.post("/adapters/dry_run", json={"user_id":"demo-user","kind":"db.pg.backup","params":params})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["ok"] and "pg_dump -d postgres" in j["cmd"] and " -f /tmp/app.sql" in j["cmd"]