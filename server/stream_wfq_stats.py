# server/stream_wfq_stats.py
# כלי עזר לסטטיסטיקות מתוך WFQBroker — בלי לשנות את המחלקה עצמה.
from __future__ import annotations
from typing import Dict, Any
from .stream_wfq import BROKER

def broker_stats() -> Dict[str, Any]:
    """
    מחזיר סטטיסטיקת תורים לכל topic:
      - queue_size
      - rate params (best-effort)
      - weight
    """
    out = {}
    # ניגש לשדות הציבוריים כפי שהוגדרו במחלקה
    for topic, pq in getattr(BROKER, "q", {}).items():
        try:
            qsize = pq.qsize()
        except Exception:
            qsize = None
        out[topic] = {
            "queue_size": qsize,
            "weight": BROKER.weights.get(topic, 1),
            "topic_tokens": BROKER.topic_tokens.get(topic, 0.0),
            "topic_rate": BROKER.topic_tb.get(topic).rate if topic in BROKER.topic_tb else None,
            "topic_burst": BROKER.topic_tb.get(topic).capacity if topic in BROKER.topic_tb else None,
        }
    out["_global"] = {
        "global_tokens": BROKER.global_tb.tokens,
        "global_rate": BROKER.global_tb.rate,
        "global_burst": BROKER.global_tb.capacity,
    }
    return out