# tests/test_http_ui_live.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import threading, time
from http.api import serve
from ui_dsl.runtime_live import StreamBroker, UILiveRuntime

def test_http_and_ui_streams():
    srv = threading.Thread(target=serve, kwargs={"host":"127.0.0.1","port":8082}, daemon=True)
    srv.start()
    time.sleep(0.3)

    broker = StreamBroker()
    events = {"timeline":[], "progress":[], "claims":[], "perf":[]}

    broker.subscribe("timeline", lambda m: events["timeline"].append(m))
    broker.subscribe("progress", lambda m: events["progress"].append(m))
    broker.subscribe("claims",   lambda m: events["claims"].append(m))
    broker.subscribe("perf",     lambda m: events["perf"].append(m))

    stop = threading.Event()
    t = threading.Thread(target=broker.run, args=(stop,), daemon=True)
    t.start()
    ui = UILiveRuntime(broker)
    data = ui.run_pipeline_and_stream(url="http://127.0.0.1:8082/run_pipeline")

    stop.set(); t.join(timeout=1.0)

    assert data["ok"] is True
    assert events["progress"] and events["timeline"]
    assert events["claims"] and events["perf"]