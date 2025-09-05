# engine/progress.py
from __future__ import annotations
import os
from typing import Dict, Any, Optional
import threading, time
from streams.backpressure import BackPressureBus, TopicPolicy
from streams.broker_client import publish_sync
from adapters.contracts.base import record_event, ResourceRequired

BUS = BackPressureBus(global_burst=2000)

# Default policies
BUS.set_policy("progress", TopicPolicy(rate_per_sec=50, burst=200, priority=2))
BUS.set_policy("timeline", TopicPolicy(rate_per_sec=20, burst=80, priority=1))     # timeline has higher prio (1<2)
BUS.set_policy("logs", TopicPolicy(rate_per_sec=200, burst=400, priority=9))       # logs lowest priority
BUS.set_policy("metrics", TopicPolicy(rate_per_sec=20, burst=80, priority=3))

class ProgressEmitter:
    def __init__(self, broker_url: Optional[str] = None):
        self.broker_url = broker_url

    def emit(self, topic: str, payload: Dict[str, Any]):
        BUS.offer(topic, payload)

    def start(self):
        def worker():
            while True:
                taken = BUS.take()
                if not taken: continue
                topic, payload = taken
                try:
                    if self.broker_url:
                        publish_sync(self.broker_url, topic, payload)
                    record_event("stream_emit", {"topic": topic, "payload": payload})
                except ResourceRequired as r:
                    # log once per session; still record locally
                    record_event("resource_required", {"resource": r.resource, "why": r.why, "install": r.how_to_install})
                except Exception as e:
                    record_event("stream_emit_error", {"topic": topic, "err": str(e)})
        t = threading.Thread(target=worker, daemon=True); t.start()

EMITTER = ProgressEmitter(broker_url=os.environ.get("IMU_BROKER_URL"))
EMITTER.start()