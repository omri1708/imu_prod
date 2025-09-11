# tests/test_generated_queue-kafka-produce.py
from fastapi.testclient import TestClient
from server.http_api import APP
client = TestClient(APP)

def test_kafka_produce_dryrun():
    params={"bootstrap":"localhost:9092","topic":"events","props_opt":" --producer-property linger.ms=5","input_opt":""}
    r=client.post("/adapters/dry_run", json={"user_id":"demo-user","kind":"queue.kafka.produce","params":params})
    assert r.status_code==200
    j=r.json(); assert j["ok"] and "--topic events" in j["cmd"]