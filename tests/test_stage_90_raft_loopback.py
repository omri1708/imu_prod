# imu_repo/tests/test_stage90_raft_loopback.py
from __future__ import annotations
import asyncio
from grounded.claims import current
from engine.user_scope import user_scope
from engine.config import load_config, save_config
from engine.caps_distributed import distributed_kv_put_capability
from distributed.raft import Cluster

def _config():
    cfg = load_config()
    cfg["guard"] = {"min_trust": 0.0, "max_age_s": 3600.0, "min_count": 0, "required_kinds": []}
    cfg["phi"] = {"max_allowed": 200.0, "per_capability_cost": {"distributed.raft.kv_put": 3.0}}
    cfg["async"] = {"max_global": 8, "per_user": 4, "per_capability": {}, "per_capability_rps": {}}
    save_config(cfg)

def test_raft_puts_and_convergence():
    _config()
    current().reset()
    with user_scope("zoe"):
        cap = distributed_kv_put_capability("zoe")
        loop = asyncio.get_event_loop()
        # הרץ כמה פקודות רצופות
        keys = [("k1","v1"), ("k2","v2"), ("k1","v3"), ("k3","v9")]
        outs = [loop.run_until_complete(cap({"key": k, "val": v})) for k,v in keys]
        assert all("put_ok" in o["text"] for o in outs), outs
        evs = current().snapshot()
        assert any(e["kind"] == "raft_elected_leader" for e in evs), "leader election missing"
        assert any(e["kind"] == "raft_commit" for e in evs), "commit evidence missing"
        # בדוק שהקומיט באמת יושם בכל הצמתים (all_equal=True בכל העדכונים)
        assert all("all_equal=True" in o["text"] for o in outs), outs

def run():
    test_raft_puts_and_convergence()
    print("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())