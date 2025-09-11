# imu_repo/tests/test_stage47_consensus_and_routing.py
from __future__ import annotations
import asyncio
from engine.realtime_and_dist import MicroRuntime

async def fake_invoke(addr: str, payload):
    # "שירות" שנכשל באחד האינסטנסים ומצליח בשני
    if addr == "bad:1":
        raise RuntimeError("down")
    return {"addr": addr, "echo": payload.get("x")}

async def main():
    rt = MicroRuntime()

    # רישום שני אינסטנסים לאותו שירות
    rt.register_service("calc", "i1", "bad:1")
    rt.register_service("calc", "i2", "good:2")

    # קלאסטר קונצנזוס בן 3 צמתים
    peers = ["n1","n2","n3"]
    for p in peers:
        n = rt.spawn_node(p, peers, lease_s=0.8)
        asyncio.create_task(n.start())

    # בחירת מנהיג והוספת רשומה עם quorum
    leader = await rt.elect_leader()
    await rt.write_consensus({"op":"set", "key":"threshold", "val": 42})

    # קריאה: router ינסה bad:1 וייפול, יעבור ל-good:2 ויצליח
    out = await rt.router.call("calc", {"x": 7}, fake_invoke)
    ok = (out["addr"] == "good:2" and out["echo"] == 7)
    return 0 if ok else 1

if __name__=="__main__":
    try:
        rc = asyncio.run(main())
        print("OK" if rc==0 else "FAIL")
        raise SystemExit(rc)
    except Exception as e:
        print("FAIL", e)
        raise SystemExit(1)