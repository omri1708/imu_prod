# imu_repo/dist/replication.py
from __future__ import annotations
from typing import Dict, Any, Callable
import asyncio, time, json, os
from dist.crdt import ORMap

STATE_ROOT = "/mnt/data/imu_repo/dist_state"

class CRDTNode:
    def __init__(self, nid: str):
        self.nid = nid
        self.state = ORMap()
        os.makedirs(STATE_ROOT, exist_ok=True)

    def apply(self, op: Dict[str,Any]) -> None:
        """
        op: {"type":"put_reg"|"add_set"|"rem_set","key":..., "value":...}
        """
        t = time.time()
        if op["type"] == "put_reg":
            self.state.put_reg(op["key"], op["value"], t)
        elif op["type"] == "add_set":
            self.state.upd_set_add(op["key"], op["value"], t)
        elif op["type"] == "rem_set":
            self.state.upd_set_rem(op["key"], op["value"], t)
        else:
            raise ValueError("unknown_op")

    def serialize(self) -> str:
        # סריאליזציה פשוטה ל-json
        d = {
            "keys_adds": list(self.state.keys.adds.items()),
            "keys_rems": list(self.state.keys.rems.items()),
            "regs": {k: (v.value, v.ts) for k,v in self.state.regs.items()},
            "sets_adds": {k: list(v.adds.items()) for k,v in self.state.sets.items()},
            "sets_rems": {k: list(v.rems.items()) for k,v in self.state.sets.items()},
        }
        return json.dumps(d)

    @staticmethod
    def deserialize(s: str) -> ORMap:
        d = json.loads(s)
        from dist.crdt import LWWSet, LWWRegister, ORMap as _OR
        st = _OR()
        st.keys.adds = {k: float(t) for k,t in d["keys_adds"]}
        st.keys.rems = {k: float(t) for k,t in d["keys_rems"]}
        for k, (val, ts) in d["regs"].items():
            st.regs[k] = LWWRegister(val, float(ts))
        for k, arr in d["sets_adds"].items():
            from dist.crdt import LWWSet
            sset = st.sets.get(k) or LWWSet()
            for v,t in arr: sset.adds[v]=float(t)
            st.sets[k]=sset
        for k, arr in d["sets_rems"].items():
            sset = st.sets.get(k) or LWWSet()
            for v,t in arr: sset.rems[v]=float(t)
            st.sets[k]=sset
        return st

    def save(self) -> None:
        with open(os.path.join(STATE_ROOT, f"{self.nid}.json"), "w", encoding="utf-8") as f:
            f.write(self.serialize())

    def load(self) -> None:
        p = os.path.join(STATE_ROOT, f"{self.nid}.json")
        if not os.path.exists(p): 
            return
        with open(p,"r",encoding="utf-8") as f:
            self.state = self.deserialize(f.read())

async def gossip_once(a: CRDTNode, b: CRDTNode) -> None:
    """חליפת מצבים דו־כיוונית ומיזוג עד עיקביות."""
    a.save(); b.save()
    a.load(); b.load()
    # החלפה:
    sa, sb = a.serialize(), b.serialize()
    a.state.merge(CRDTNode.deserialize(sb))
    b.state.merge(CRDTNode.deserialize(sa))
    a.save(); b.save()