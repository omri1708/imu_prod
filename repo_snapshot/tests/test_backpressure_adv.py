# tests/test_backpressure_adv.py
# -*- coding: utf-8 -*-
import asyncio, pytest
from realtime.backpressure import GlobalTokenBucket
from realtime.priority_bus import AsyncPriorityTopicBus

@pytest.mark.asyncio
async def test_fairness_basic():
    bucket = GlobalTokenBucket(50, 200.0)
    bus = AsyncPriorityTopicBus(bucket, per_topic_rates={"A": (10, 50.0), "B": (10, 50.0)})
    await bus.start()

    gotA, gotB = [], []

    async def sub(topic, sink):
        async for m in bus.subscribe(topic):
            sink.append(m)
            if len(sink) >= 5: break

    t1 = asyncio.create_task(sub("A", gotA))
    t2 = asyncio.create_task(sub("B", gotB))

    for i in range(10):
        await bus.publish("A", f"A{i}", priority=5)
    for i in range(10):
        await bus.publish("B", f"B{i}", priority=5)

    await asyncio.wait_for(t1, 2.0)
    await asyncio.wait_for(t2, 2.0)
    assert len(gotA) == 5 and len(gotB) == 5  # הוגנות בסיסית