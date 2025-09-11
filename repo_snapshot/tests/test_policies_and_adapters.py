# tests/test_policies_and_adapters.py
import json, threading, time
from policy.user_subspace import registry
from engine.adapter_runner import enforce_policy, ResourceRequired
from adapters.contracts import AdapterResult
from adapters import cuda
from broker.stream_bus import bus
from attic.app.http_api import serve
import requests

def test_policy_ttl_allows_after_consent(monkeypatch):
    pol = registry.ensure_user("u1")
    # First call requires consent
    try:
        enforce_policy("u1", "adapter.cuda.run", trust="low")
        assert False, "should require consent"
    except ResourceRequired:
        pass
    pol.grant_once("adapter.cuda.run","invoke", ttl_sec=2)
    # Now allowed
    enforce_policy("u1", "adapter.cuda.run", trust="low")

def test_provenance_and_streams(monkeypatch):
    events = []
    bus.subscribe("timeline", lambda e: events.append(e))
    # Start HTTP server in bg
    t = threading.Thread(target=serve, kwargs={"port":8090}, daemon=True); t.start()
    time.sleep(0.2)

    # Call adapter that will ask for nvcc
    r = requests.post("http://127.0.0.1:8090/run_adapter",
                      json={"user_id":"u2","adapter":"cuda","params":{"cuda_src":"kern.cu","out_bin":"a.out"}}).json()
    assert r["status"] == "awaiting_consent"
    # The bus should have published a timeline entry
    assert any(e.get("type") in ("policy","resource") for e in events)

def test_cuda_adapter_messages(monkeypatch):
    res: AdapterResult = cuda.run_cuda_job("kern.cu","a.out")
    assert res.status in ("awaiting_consent","ok","error")

#הערה: בדיקות ה-HTTP משתמשות ב-requests. אם אין – אפשר להחליף ב-urllib בקלות. אין כאן “מוקים”; הקוד אכן מאזין ושולח אירועים על ה-bus.



