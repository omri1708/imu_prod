# imu_repo/engine/gates/distributed_gate.py
from __future__ import annotations
from typing import Dict, Any
from dist.raft_lite import cluster_health

class DistributedGate:
    """
    מחייב רוב (quorum) נוכחי + מנהיג חי לפני פריסה/ריספונד.
    """
    def __init__(self, require_quorum: bool=True, require_leader: bool=True):
        self.require_quorum = bool(require_quorum)
        self.require_leader = bool(require_leader)

    def check(self) -> Dict[str,Any]:
        h = cluster_health()
        ok = True
        viol=[]
        if self.require_quorum and not h.get("quorum_ok",False):
            ok=False; viol.append(("no_quorum", h))
        if self.require_leader and not h.get("leader"):
            ok=False; viol.append(("no_leader", h))
        return {"ok": ok, "violations": viol, "health": h}