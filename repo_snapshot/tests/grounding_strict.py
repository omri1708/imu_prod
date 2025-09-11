# imu_repo/tests/grounding_strict.py
from __future__ import annotations
from typing import Dict, Any, List
from engine.pipeline import Engine

def prog_without_evidence() -> List[Dict[str,Any]]:
    return [
        {"op":"PUSH","value":42},
        {"op":"STORE","reg":"x"},
        {"op":"RESPOND","status":200,"body":{"x":"reg:x"}}
    ]

def prog_with_evidence() -> List[Dict[str,Any]]:
    return [
        {"op":"PUSH","value":13},
        {"op":"PUSH","value":29},
        {"op":"ADD"},
        {"op":"STORE","reg":"s"},
        {"op":"EVIDENCE","claim":"sum_result","sources":["bench:sum"]},
        {"op":"RESPOND","status":200,"body":{"sum":"reg:s"}}
    ]

def run():
    e = Engine(mode="strict")
    c1,b1 = e.run_program(prog_without_evidence(), {}, policy="strict")
    print("no_evidence:", c1, b1)  # צפוי 412
    c2,b2 = e.run_program(prog_with_evidence(), {}, policy="strict")
    print("with_evidence:", c2, b2)  # צפוי 200
    return 0 if (c1==412 and c2==200) else 1

if __name__=="__main__":
    raise SystemExit(run())