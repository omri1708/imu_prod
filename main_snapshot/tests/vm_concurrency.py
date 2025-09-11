# imu_repo/tests/vm_concurrency.py
from __future__ import annotations
from typing import List, Dict, Any
from core.vm.vm import VM, Limits

def subtask(a: float, b: float) -> List[Dict[str,Any]]:
    return [
        {"op":"PUSH","value":a},
        {"op":"PUSH","value":b},
        {"op":"ADD"},
        {"op":"STORE","reg":"s"},
        {"op":"SLEEP_MS","ms":50},
        {"op":"EVIDENCE","claim":"sum_result","sources":["bench:sum"]},
        {"op":"RESPOND","status":200,"body":{"sum":"reg:s"}}
    ]

def program() -> List[Dict[str,Any]]:
    return [
        {"op":"SPAWN","body":subtask(10,32),"as":"t1"},
        {"op":"SPAWN","body":subtask(7,5),"as":"t2"},
        {"op":"JOIN","task":"reg:t1","timeout_s":2},
        {"op":"STORE","reg":"r1"},
        {"op":"JOIN","task":"reg:t2","timeout_s":2},
        {"op":"STORE","reg":"r2"},
        {"op":"EVIDENCE","claim":"sum_result","sources":["bench:sum"]},
        {"op":"RESPOND","status":200,"body":{"A":"reg:r1","B":"reg:r2"}}
    ]

def run():
    vm = VM(Limits(max_async_tasks=8, max_sleep_ms=200))
    c,b,_ = vm.run(program(), {})
    print("concurrency:", c, b)
    ok = (c==200 and b.get("A",{}).get("ok") and b.get("B",{}).get("ok"))
    return 0 if ok else 1

if __name__=="__main__":
    raise SystemExit(run())