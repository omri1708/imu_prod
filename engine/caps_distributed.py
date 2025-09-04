# imu_repo/engine/caps_distributed.py
from __future__ import annotations
import asyncio
from typing import Dict, Any
from grounded.claims import current
from engine.capability_wrap import text_capability_for_user
from distributed.raft import Cluster

# מחזיקים קלאסטר יחיד בזיכרון (לבדיקות/לוקאל)
_CLUSTER: Cluster | None = None

async def _ensure_cluster() -> Cluster:
    global _CLUSTER
    if _CLUSTER is None:
        _CLUSTER = Cluster(n=3)
        await _CLUSTER.start()
    return _CLUSTER

async def _kv_put_impl(payload: Dict[str,Any]) -> str:
    key = str(payload.get("key"))
    val = str(payload.get("val"))
    c = await _ensure_cluster()
    # המתן לבחירת מנהיג
    leader = await c.wait_for_leader(timeout_s=2.5)
    if not leader:
        return "[FALLBACK] no_leader"
    ok = await leader.propose_put(key, val)
    if not ok:
        return "[FALLBACK] put_failed"
    # בדיקה שהמדינות זהות (קומיט)
    states = [n.kv.copy() for n in c.nodes]
    all_eq = all(states[0] == s for s in states[1:])
    current().add_evidence("raft_put_ok", {
        "source_url":"raft://cluster/local","trust":0.97,"ttl_s":600,
        "payload":{"key":key,"val":val,"all_equal":all_eq}
    })
    return f"put_ok key={key} val={val} all_equal={all_eq}"

def distributed_kv_put_capability(user_id: str):
    """
    עוטף כיכולת טקסטואלית עם Φ/Guard/Async-Caps (מהשלבים הקודמים).
    """
    return text_capability_for_user(_kv_put_impl, user_id=user_id,
                                    capability_name="distributed.raft.kv_put",
                                    cost=3.0)