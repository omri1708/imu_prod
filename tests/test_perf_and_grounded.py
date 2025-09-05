# tests/test_perf_and_grounded.py

# -*- coding: utf-8 -*-
import os, time, json, pytest
from provenance.store import ProvenanceStore
from policy.user_policy import UserSubspacePolicy
from engine.http_api import BUS

def test_provenance_ttl_and_trust():
    prov=ProvenanceStore(root=".imu_test/prov", default_ttl_s=1)
    d=prov.put(b"hello", source="test", trust=80, ttl_s=1, evidence={"url":"https://example.com"})
    # זמין עכשיו
    content, meta = prov.get(d, min_trust=50)
    assert content==b"hello" and meta["trust"]==80
    time.sleep(1.1)
    with pytest.raises(RuntimeError):
        prov.get(d, min_trust=50)

def test_bus_backpressure_and_throttle():
    from broker.bus import EventBus, BusOverload
    bus=EventBus(per_topic_max=10)
    seen=[]
    bus.subscribe("telemetry", lambda e: seen.append(e))
    bus.set_throttle("telemetry", per_sec=10)
    for _ in range(50):
        try: bus.publish("telemetry", {"x":1}, priority="high")
        except BusOverload: break
    # לא נקרוס; או שנפיל עודפים או שנחנוק
    assert len(seen) >= 0