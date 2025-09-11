# tests/test_generated_queue-nats-publish.py
from fastapi.testclient import TestClient
from server.http_api import APP
client = TestClient(APP)

def test_nats_publish_dryrun():
    params={"server":"nats://127.0.0.1:4222","subject":"events","message":"hello","creds_opt":"","tls_opt":""}
    r=client.post("/adapters/dry_run", json={"user_id":"demo-user","kind":"queue.nats.publish","params":params})
    assert r.status_code==200
    j=r.json(); assert j["ok"] and 'nats pub events "hello"' in j["cmd"]