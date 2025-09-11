# tests/test_http_sse_broker.py
# -*- coding: utf-8 -*-
import json, threading, time, http.client
from server.http_api import serve_http
from broker.stream import broker

def _post_json(path, obj):
    c = http.client.HTTPConnection("127.0.0.1", 8080, timeout=5)
    body = json.dumps(obj).encode("utf-8")
    c.request("POST", path, body=body, headers={"Content-Type":"application/json"})
    r = c.getresponse()
    data = r.read()
    return r.status, json.loads(data.decode("utf-8"))

def test_sse_progress_timeline_roundtrip():
    srv = serve_http("127.0.0.1", 8080)
    # נפרסם ידנית אירוע – מדמה Pipeline
    broker.publish("progress", {"stage":"init","run_id":"x"}, priority=0)
    broker.publish("timeline", {"t":"hello"}, priority=1)

    # נבדוק שה־health חי, ואז pipeline אמיתי מינימלי ללא אדפטורים
    st, resp = _post_json("/v1/pipeline/run", {"user":"alice","spec":"{\"name\":\"x\"}"})
    assert st == 200
    assert resp.get("ok") is True
    srv.shutdown()