# tests/test_qos_rate_limit.py
# -*- coding: utf-8 -*-
import pytest
from realtime.qos_broker import start, publish
from contracts.errors import RateLimitExceeded

def test_qos_limits():
    q = start("127.0.0.1", 8799, global_rate=5, global_burst=5, per_topic_rate=2, per_topic_burst=2, max_queue=10)
    ok = 0; fail = 0
    for i in range(20):
        try:
            publish("progress/t1", {"value": i}, priority=1); ok += 1
        except RateLimitExceeded:
            fail += 1
    assert ok > 0 and fail > 0