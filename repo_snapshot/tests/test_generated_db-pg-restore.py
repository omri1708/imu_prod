# tests/test_generated_db-pg-restore.py
from fastapi.testclient import TestClient
from server.http_api import APP
client = TestClient(APP)

def test_pg_restore_dryrun():
    params={"db_url":"postgres://user:pass@localhost/app","in":"./dump.tar","clean_opt":" --clean","ifexists_opt":" --if-exists","jobs_opt":" -j 2","extra_opt":""}
    r=client.post("/adapters/dry_run", json={"user_id":"demo-user","kind":"db.pg.restore","params":params})
    assert r.status_code==200
    j=r.json(); assert j["ok"] and "pg_restore -d postgres://" in j["cmd"]
