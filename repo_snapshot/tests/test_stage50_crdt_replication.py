# imu_repo/tests/test_stage50_crdt_replication.py
from __future__ import annotations
import asyncio
from dist.replication import CRDTNode, gossip_once

async def main():
    a = CRDTNode("nA"); b = CRDTNode("nB")
    a.apply({"type":"put_reg","key":"version","value":"1.0"})
    b.apply({"type":"add_set","key":"features","value":"realtime"})
    await gossip_once(a,b)
    # עדכונים משני הצדדים → עוד gossip
    a.apply({"type":"add_set","key":"features","value":"crdt"})
    b.apply({"type":"put_reg","key":"version","value":"1.1"})
    await gossip_once(a,b)
    ok = (a.state.get_reg("version")==b.state.get_reg("version")=="1.1") \
         and ("realtime" in a.state.get_set("features") and "crdt" in b.state.get_set("features"))
    print("OK" if ok else "FAIL")
    return 0 if ok else 1

if __name__=="__main__":
    raise SystemExit(asyncio.run(main()))