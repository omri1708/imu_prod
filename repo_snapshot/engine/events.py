# engine/events.py
# -*- coding: utf-8 -*-
from stream.broker import BROKER

TOPIC_PROGRESS = "pipeline.progress"
TOPIC_TIMELINE = "pipeline.timeline"

def emit_progress(stage: str, user: str, **kw):
    BROKER.publish(TOPIC_PROGRESS, {"stage": stage, "user": user, **kw}, prio=8)

def emit_timeline(event: str, user: str, **kw):
    BROKER.publish(TOPIC_TIMELINE, {"event": event, "user": user, **kw}, prio=6)
