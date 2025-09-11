# imu_repo/tests/test_stage60_distributed.py
from __future__ import annotations
import os, time, json, random, multiprocessing as mp

from dist.raft_lite import ensure_node, current_leader, cluster_health, append_record_if_leader
from dist.job_queue import enqueue, reserve, ack, nack, replay_from_wal
from dist.worker import start_pool, join_pool
from engine.gates.distributed_gate import DistributedGate

CL = "/mnt/data/imu_repo/cluster"
Q  = "/mnt/data/imu_repo/queue"

def assert_true(b, msg=""):
    if not b:
        print("ASSERT FAIL:", msg)
        raise SystemExit(1)

# ---- helpers ----
def cleanup():
    for p in (CL,Q):
        if os.path.exists(p):
            for root,dirs,files in os.walk(p, topdown=False):
                for f in files:
                    os.remove(os.path.join(root,f))
                for d in dirs:
                    os.rmdir(os.path.join(root,d))

def sample_task(payload):
    """
    payload = {"a":int,"b":int,"out":"/mnt/data/imu_repo/out_X.json","fail":False}
    """
    a = int(payload["a"]); b = int(payload["b"])
    outp = payload["out"]
    if payload.get("fail"):
        # כתיבה ואז כשל -> נבדוק rollback delete_file
        open(outp,"w",encoding="utf-8").write("PARTIAL")
        return (False, {"error":"forced_failure"}, {"type":"delete_file","path":outp})
    res = {"sum": a+b, "mul": a*b}
    with open(outp,"w",encoding="utf-8") as f:
        json.dump(res, f)
    return (True, {"wrote": outp}, None)

def wait_for_leader(timeout_s=5.0):
    t0=time.time()
    while time.time()-t0 < timeout_s:
        lid=current_leader()
        if lid: return lid
        time.sleep(0.1)
    return None

def test_cluster_and_gate():
    cleanup()
    n1 = ensure_node("001")
    n2 = ensure_node("002")
    n3 = ensure_node("003")
    lid = wait_for_leader()
    assert_true(lid in ("001","002","003"), "no leader elected")
    # Append רק אם מנהיג
    ok = append_record_if_leader({"msg":"hello"})
    assert_true(ok, "append failed (no leader?)")
    # Gate
    g = DistributedGate(require_quorum=True, require_leader=True)
    res = g.check()
    assert_true(res["ok"] and res["health"]["quorum_ok"])

    # עצור
    for p in (n1,n2,n3):
        p.terminate(); p.join()

def test_queue_exactly_once_like():
    cleanup()
    # Enqueue 8 עבודות, אחת עם fail= True
    outs=[]
    for i in range(8):
        fail = (i==3)
        outp = f"/mnt/data/imu_repo/out_{i}.json"
        outs.append(outp)
        r = enqueue({"a":i,"b":i+1,"out":outp,"fail":fail})
        assert_true(r["ok"])
    # הפעל שני workers
    procs = start_pool(2, sample_task)
    join_pool(procs)

    # ודא תוצאות
    done_dir = os.path.join(Q,"done")
    failed_dir = os.path.join(Q,"failed")
    dones = set(fn[:-5] for fn in os.listdir(done_dir))
    fails = set(fn[:-5] for fn in os.listdir(failed_dir))
    assert_true(len(dones)+len(fails) == 8, "not all jobs finished")

    # קובץ של הכשל צריך להימחק (rollback)
    assert_true(not os.path.exists(outs[3]), "rollback did not delete partial file")

    # בדוק שאין כפילויות בקבצי out
    seen=set()
    for i,outp in enumerate(outs):
        if i==3:  # נכשל — אין קובץ
            assert_true(not os.path.exists(outp)); continue
        assert_true(os.path.exists(outp), f"missing result {outp}")
        s = json.dumps(json.load(open(outp)))
        assert_true(s not in seen, "duplicate result?")  # בדיקה רופפת
        seen.add(s)

def test_replay_from_wal():
    # השבת מצב מאפס
    stats = replay_from_wal(clear_first=True)
    # לאחר replay יהיו עבודות במצבן האחרון; אין וורקרים כעת — רק בודקים שלא קרס
    assert_true(isinstance(stats, dict))
    print("WAL stats:", stats)

def run():
    test_cluster_and_gate()
    test_queue_exactly_once_like()
    test_replay_from_wal()
    print("OK")
    return 0

if __name__=="__main__":
    raise SystemExit(run())