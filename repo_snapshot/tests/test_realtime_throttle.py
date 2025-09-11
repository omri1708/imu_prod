# tests/test_realtime_throttle.py
# -*- coding: utf-8 -*-
import asyncio
import pytest
from realtime.backpressure import GlobalTokenBucket, RateLimitExceeded
from realtime.priority_bus import AsyncPriorityTopicBus

@pytest.mark.asyncio
async def test_global_backpressure_blocks_burst():
    bucket = GlobalTokenBucket(capacity=5, rate_tokens_per_sec=1.0)
    bus = AsyncPriorityTopicBus(bucket, per_topic_rates={"logic": (5, 5.0)})
    # חמש הודעות נכנסות; השישית נחסמת
    for _ in range(5):
        ok = await bus.publish("logic", {"x": 1}, priority=1)
        assert ok
    with pytest.raises(RateLimitExceeded):
        await bus.publish("logic", {"x": 2}, priority=1, cost_tokens=1)

@pytest.mark.asyncio
async def test_priority_drop_when_full():
    bucket = GlobalTokenBucket(capacity=100, rate_tokens_per_sec=100.0)
    bus = AsyncPriorityTopicBus(bucket, per_topic_rates={"logs": (50, 50.0)}, max_queue_size_per_topic=2)
    # נמלא תור; נשלח עוד אחת עם עדיפות נמוכה ותיפול (drop)
    await bus.publish("logs", "A", priority=100)
    await bus.publish("logs", "B", priority=50)
    ok = await bus.publish("logs", "C", priority=999, drop_if_full=True)
    assert ok is False