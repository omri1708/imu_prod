# tests/test_generated_storage-gcs-sync.py
from fastapi.testclient import TestClient
from server.http_api import APP
client = TestClient(APP)

def test_gcs_sync_dryrun():
    params={"src":"./out","dst":"gs://bucket/path","opts_opt":" -d"}
    r=client.post("/adapters/dry_run", json={"user_id":"demo-user","kind":"storage.gcs.sync","params":params})
    assert r.status_code==200
    j=r.json(); assert j["ok"] and "gsutil -m rsync -r ./out gs://bucket/path -d" in j["cmd"]