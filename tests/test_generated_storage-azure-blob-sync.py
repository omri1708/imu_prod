# tests/test_generated_storage-azure-blob-sync.py
from fastapi.testclient import TestClient
from server.http_api import APP
client=TestClient(APP)

def test_azure_sync_dryrun():
    params={"src":"./out","dst":"https://account.blob.core.windows.net/container/path","sas_opt":"","extra_opt":" --delete-destination=true"}
    r=client.post("/adapters/dry_run", json={"user_id":"demo-user","kind":"storage.azure.blob.sync","params":params})
    assert r.status_code==200
    j=r.json(); assert j["ok"] and "azcopy sync ./out https://account.blob" in j["cmd"]
