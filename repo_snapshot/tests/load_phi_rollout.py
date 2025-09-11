# imu_repo/tests/load_phi_rollout.py
from __future__ import annotations
import time, random
from typing import Dict, Any
from engine.pipeline import Engine
from governance.ab_verify import ABVerifier
from governance.canary_rollout import CanaryRollout
from optimizer.phi import compute_phi
from obs.kpi import KPI

def prog_fast():
    return [
        {"op":"PUSH","value":1},{"op":"PUSH","value":2},{"op":"ADD"},{"op":"STORE","reg":"s"},
        {"op":"EVIDENCE","claim":"sum_result","sources":["bench:sum"]},
        {"op":"RESPOND","status":200,"body":{"sum":"reg:s"}}
    ]

def prog_slow():
    return [
        {"op":"PUSH","value":1},{"op":"PUSH","value":2},{"op":"ADD"},{"op":"STORE","reg":"s"},
        {"op":"SLEEP_MS","ms":20},
        {"op":"EVIDENCE","claim":"sum_result","sources":["bench:sum"]},
        {"op":"RESPOND","status":200,"body":{"sum":"reg:s"}}
    ]

def run_once(engine: Engine, program):
    t0=time.time()
    code, body = engine.run_program(program, {}, policy="strict")
    lat=(time.time()-t0)*1000
    ok=(code==200)
    engine.kpi.record(lat, not ok)
    return lat, ok

def run():
    # baseline: fast, candidate: slow (נדחה)
    baseline = Engine(mode="strict")
    candidate = Engine(mode="strict")

    # A/B
    ab = ABVerifier()
    for i in range(100):
        # חצי־חצי
        run_once(baseline, prog_fast())
        run_once(candidate, prog_slow())

    k_base = baseline.kpi.snapshot(); k_cand = candidate.kpi.snapshot()
    phi_base = compute_phi({"p95":k_base["p95"],"latency_ms":k_base["avg"],"error":k_base["error_rate"]>0})
    phi_cand = compute_phi({"p95":k_cand["p95"],"latency_ms":k_cand["avg"],"error":k_cand["error_rate"]>0})
    print("Φ base:",phi_base," Φ cand:",phi_cand)

    # קנרי: יאשר רק אם Ф_candidate < Ф_baseline
    can = CanaryRollout()
    decision = (phi_cand < phi_base)
    print("promote?", decision)
    return 0 if not decision else 1

if __name__=="__main__":
    raise SystemExit(run())