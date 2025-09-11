# ui_dsl/runtime_live.py (חיבור חי לברוקר/HTTP לעדכון progress/timeline)
# -*- coding: utf-8 -*-
from __future__ import annotations
import json, threading, time, queue, urllib.request
from typing import Dict, Any, Callable

class StreamBroker:
    """ברוקר פשוט בתוך התהליך (אפשר להחליף ב-NATS/Kafka)."""
    def __init__(self):
        self.subs: Dict[str, Callable[[Dict[str,Any]],None]] = {}
        self.q = queue.Queue()

    def publish(self, topic:str, msg:Dict[str,Any]):
        self.q.put((topic, msg))

    def subscribe(self, topic:str, cb:Callable[[Dict[str,Any]],None]):
        self.subs[topic] = cb

    def run(self, stop):
        while not stop.is_set():
            try:
                topic, msg = self.q.get(timeout=0.25)
                if topic in self.subs:
                    self.subs[topic](msg)
            except queue.Empty:
                pass

class UILiveRuntime:
    """
    “DSL ל-UI”: התחברות ל-/run_pipeline, הזרמת אירועי timeline/progress.
    """
    def __init__(self, broker:StreamBroker):
        self.broker = broker

    def run_pipeline_and_stream(self, url="http://127.0.0.1:8081/run_pipeline"):
        # סימון התחלה
        self.broker.publish("timeline", {"t":"start","msg":"Pipeline started"})
        req = urllib.request.Request(url, method="POST", data=b"{}", headers={"Content-Type":"application/json"})
        try:
            rsp = urllib.request.urlopen(req, timeout=8)
            data = json.loads(rsp.read().decode())
            self.broker.publish("progress", {"pct":100})
            self.broker.publish("timeline", {"t":"end","msg":"Pipeline finished","ok":data.get("ok",False)})
            self.broker.publish("claims", {"claims": data.get("result",{}).get("claims",[])})
            self.broker.publish("perf", {"perf": data.get("perf",{})})
            return data
        except Exception as e:
            self.broker.publish("timeline", {"t":"error","msg":str(e)})
            raise