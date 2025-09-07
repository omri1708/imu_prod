# server/stream_gateway.py
# FastAPI router: publish/poll events into WFQ Broker with server-side throttling.
from __future__ import annotations
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import time

from server.stream_wfq import BROKER

router = APIRouter(prefix="/events", tags=["events"])

class PublishEvent(BaseModel):
    topic: str = Field(..., min_length=1)
    producer: str = Field("api", min_length=1, max_length=64)
    priority: int = Field(5, ge=0, le=10)
    event: Dict[str, Any] = Field(default_factory=dict)

@router.post("/publish")
def publish(ev: PublishEvent):
    # ensure topic exists with safe defaults
    BROKER.ensure_topic(ev.topic, rate=100.0, burst=500, weight=2 if ev.topic=="timeline" else 1)
    e = dict(ev.event)
    e.setdefault("type", "event")
    e.setdefault("ts", time.time())
    ok = BROKER.submit(ev.topic, ev.producer, e, priority=ev.priority)
    if not ok:
        return {"ok": False, "reason": "rate_limited_or_dropped"}
    return {"ok": True}

@router.get("/poll")
def poll(topic: str = Query(..., min_length=1), max_items: int = Query(100, ge=1, le=1000)):
    BROKER.ensure_topic(topic, rate=100.0, burst=500, weight=2 if topic=="timeline" else 1)
    batch = BROKER.poll(topic, max_items=max_items)
    return {"ok": True, "events": batch}

class ProgressIn(BaseModel):
    topic: str = "timeline"
    producer: str = "api"
    pct: float = Field(..., ge=0, le=100)
    note: Optional[str] = None

@router.post("/progress")
def progress(pr: ProgressIn):
    BROKER.ensure_topic(pr.topic, rate=100.0, burst=500, weight=3)
    ev = {"type": "progress", "ts": time.time(), "pct": pr.pct, "note": pr.note}
    ok = BROKER.submit(pr.topic, pr.producer, ev, priority=3)
    return {"ok": ok}