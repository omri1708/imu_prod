# imu_repo/tests/test_stage54_db_queue_caps.py
from __future__ import annotations
from caps.queue import FileQueue
from caps.sqlite_sandbox import execute, db_path
import os

def run():
    q = FileQueue("/mnt/data/imu_repo/queues/demo")
    name = q.enqueue({"task":"sum","a":2,"b":5})
    msg = q.dequeue()
    ok1 = (msg and msg["task"]=="sum")
    q.ack(msg)

    # SQLite
    db = "demo_db"
    p = db_path(db)
    if os.path.exists(p): os.unlink(p)
    execute(db, "CREATE TABLE items(id INTEGER PRIMARY KEY, name TEXT)")
    execute(db, "INSERT INTO items(name) VALUES (?)", ("alpha",))
    execute(db, "INSERT INTO items(name) VALUES (?)", ("beta",))
    rows = execute(db, "SELECT id, name FROM items ORDER BY id")
    ok2 = (len(rows)==2 and rows[0][1]=="alpha" and rows[1][1]=="beta")

    print("OK" if (ok1 and ok2) else "FAIL")
    return 0 if (ok1 and ok2) else 1

if __name__=="__main__":
    raise SystemExit(run())