# tests/test_generated_queue-redis-subscribe.py
from fastapi.testclient import TestClient
from server.http_api import APP

client = TestClient(APP)

def test_redis_subscribe_dryrun():
    params = {"host":"127.0.0.1","port":6379,"channel":"events","auth_opt":"","db_opt":""}
    r = client.post("/adapters/dry_run", json={"user_id":"demo-user","kind":"queue.redis.subscribe","params":params})
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] and "redis-cli -h 127.0.0.1 -p 6379" in j["cmd"] and " SUBSCRIBE events" in j["cmd"]
