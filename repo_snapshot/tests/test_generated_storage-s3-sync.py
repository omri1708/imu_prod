# tests/test_generated_storage-s3-sync.py
from fastapi.testclient import TestClient
from server.http_api import APP

client = TestClient(APP)

def test_s3_sync_dryrun():
    params = {
        "src":"./out",
        "dst":"s3://my-bucket/backup",
        "profile_opt":" --profile default",
        "region_opt":" --region us-east-1",
        "extra_opt":" --delete"
    }
    r = client.post("/adapters/dry_run", json={"user_id":"demo-user","kind":"storage.s3.sync","params":params})
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] and "aws s3 sync ./out s3://my-bucket/backup" in j["cmd"] and "--profile default" in j["cmd"]