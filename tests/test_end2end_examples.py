# tests/test_end2end_examples.py
# -*- coding: utf-8 -*-
import shutil, pytest, json
from grounded.http_verifier import http_head_exists
from realtime.backpressure import GlobalTokenBucket
from realtime.priority_bus import AsyncPriorityTopicBus
import asyncio

def test_http_verifier_smoke():
    ok = http_head_exists("https://example.org/", timeout_sec=3.0)
    # אם אין רשת בסביבה → נוותר (skip)
    if ok is None:
        pytest.skip("no network or blocked")
    assert ok in (True, False)

@pytest.mark.asyncio
async def test_priority_bus_basic():
    bucket = GlobalTokenBucket(10, 100.0)
    bus = AsyncPriorityTopicBus(bucket, per_topic_rates={"logic": (5, 50.0)})
    got = []
    async def consumer():
        async for p in bus.subscribe("logic"):
            got.append(p)
            if len(got) >= 2: break
    task = asyncio.create_task(consumer())
    await bus.publish("logic", "A", priority=1)
    await bus.publish("logic", "B", priority=5)
    await asyncio.wait_for(task, timeout=1.0)
    assert got == ["A", "B"]