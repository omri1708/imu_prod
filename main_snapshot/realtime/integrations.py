# realtime/integrations.py (Hook ל-HTTP API)
# -*- coding: utf-8 -*-
from realtime import qos_broker

def start_realtime(host="127.0.0.1", ws_port=8766):
    qos_broker.start(host=host, port=ws_port,
                     global_rate=500, global_burst=800,
                     per_topic_rate=120, per_topic_burst=240, max_queue=20000)

def push_progress(id_: str, value: int):
    qos_broker.publish(f"progress/{id_}", {"value": int(value)}, priority=1)

def push_timeline(stream: str, event: dict, priority: int = 5):
    qos_broker.publish(f"timeline/{stream}", {"event": event}, priority=priority)