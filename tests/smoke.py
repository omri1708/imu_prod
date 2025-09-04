# imu_repo/tests/smoke.py
from __future__ import annotations
import json
from engine.pipeline import Engine

def main():
    eng = Engine()
    prog = [
        {"op":"PUSH","ref":"$.payload.a"},
        {"op":"PUSH","ref":"$.payload.b"},
        {"op":"ADD"},
        {"op":"STORE","reg":"sum"},
        {"op":"EVIDENCE","claim":"a_plus_b_equals_sum","sources":["tests:smoke"]},
        {"op":"RESPOND","status":200,"body":{"sum":"reg:sum"}}
    ]
    status, body = eng.run_program(prog, {"a": 10, "b": 32})
    print("STATUS", status)
    print(json.dumps(body, ensure_ascii=False, indent=2))
    assert status == 200
    assert body.get("sum") == 42
    assert "_provenance" in body
    print("SMOKE OK")

if __name__ == "__main__":
    main()
