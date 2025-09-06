# server/stream_policy_router.py
# Policy Routing דינמי ל-WFQ: מכוון משקל/קצב לפי עומס תורים.
from __future__ import annotations
from typing import Dict, Any
import threading, time
from .stream_wfq import BROKER
from .stream_wfq_stats import broker_stats

_ADJUST_STOP=False
_THREAD=None

def _adjust_once() -> Dict[str,Any]:
    """
    קריטריון פשוט:
      - אם queue_size(topic) > 500 → weight += 1 (עד 5), rate += 20% (תקרה 200)
      - אם queue_size(topic) < 50  → weight -= 1 (עד 1), rate -= 10% (רצפה 20)
    """
    stats=broker_stats()
    out={}
    for topic, s in stats.items():
        if topic=="_global": continue
        qsize = s.get("queue_size") or 0
        weight = BROKER.weights.get(topic, 1)
        rate_tb = BROKER.topic_tb.get(topic)
        if not rate_tb: continue
        rate = float(getattr(rate_tb,"rate",50.0))
        if qsize > 500:
            weight=min(5, weight+1)
            rate=min(200.0, rate*1.2)
        elif qsize < 50:
            weight=max(1, weight-1)
            rate=max(20.0, rate*0.9)
        BROKER.weights[topic]=weight
        rate_tb.rate = rate
        out[topic]={"weight":weight,"rate":rate,"q":qsize}
    return out

def start_policy_router(interval_s: float = 1.5):
    global _THREAD, _ADJUST_STOP
    if _THREAD and _THREAD.is_alive(): return
    _ADJUST_STOP=False
    def _loop():
        while not _ADJUST_STOP:
            try:
                _adjust_once()
            except Exception:
                pass
            time.sleep(interval_s)
    _THREAD=threading.Thread(target=_loop, name="imu-policy-router", daemon=True)
    _THREAD.start()

def stop_policy_router():
    global _ADJUST_STOP
    _ADJUST_STOP=True