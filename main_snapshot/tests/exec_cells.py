# imu_repo/tests/exec_cells.py
from __future__ import annotations
from exec.errors import ResourceRequired
from exec.cells import run_code
from exec.select import choose
from engine.exec_api import exec_best

def run():
    # Python
    py = run_code("python", 'print("hello from python")', user_id="alice", cell_name="hello")
    print(py["lang"], py["exit"], py["stdout"].strip())

    # Node (אם קיים)
    try:
        nd = run_code("node", 'console.log("hi from node")', user_id="alice", cell_name="hello")
        print(nd["lang"], nd["exit"], nd["stdout"].strip())
    except ResourceRequired as rr:
        print("REQ:", rr.how)

    # בחירה אוטומטית (לפי תגיות וזמינות)
    task = {"tags":["system","concurrency"], "code": 'print("auto on python as fallback")'}
    res = exec_best(task, ctx={"user_id":"alice","__routing_hints__":{"search_depth":"deep"}})
    print("best:", res["lang"], res["exit"])
    return 0

if __name__=="__main__":
    raise SystemExit(run())