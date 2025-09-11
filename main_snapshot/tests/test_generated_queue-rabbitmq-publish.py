# tests/test_generated_queue-rabbitmq-publish.py
from fastapi.testclient import TestClient
from server.http_api import APP
client = TestClient(APP)

def test_rabbitmq_publish_dryrun():
    params={"host":"127.0.0.1","port":15672,"exchange":"amq.direct","routing_key":"events","payload":"hello","user_opt":"","pass_opt":"","vhost_opt":"","props_opt":""}
    r=client.post("/adapters/dry_run", json={"user_id":"demo-user","kind":"queue.rabbitmq.publish","params":params})
    assert r.status_code==200
    j=r.json(); assert j["ok"] and "rabbitmqadmin -H 127.0.0.1 -P 15672" in j["cmd"] and "exchange=amq.direct" in j["cmd"]
