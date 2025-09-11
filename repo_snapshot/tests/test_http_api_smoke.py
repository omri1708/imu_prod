# tests/test_http_api_smoke.py
# -*- coding: utf-8 -*-
import threading, time, json, urllib.request
from attic.app.http_api import serve

def _start():
    t = threading.Thread(target=lambda: serve("127.0.0.1", 8089), daemon=True)
    t.start()
    time.sleep(0.2)

def test_pipeline_endpoint_smoke():
    _start()
    spec = {"name":"demo","targets":[{"path":"x.txt","content":"hi"}]}
    req = urllib.request.Request("http://127.0.0.1:8089/v1/pipeline/run",
                                 data=json.dumps({"user":"t","spec":json.dumps(spec)}).encode("utf-8"),
                                 headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        assert r.status in (200,202)
        j = json.loads(r.read().decode("utf-8"))
        assert j["ok"] is True
        job = j["job"]
    # משוך סטטוס
    with urllib.request.urlopen(f"http://127.0.0.1:8089/v1/jobs/{job}", timeout=10) as r2:
        assert r2.status == 200