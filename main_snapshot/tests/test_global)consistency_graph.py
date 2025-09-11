# imu_repo/tests/test_global_consistency_graph.py
from __future__ import annotations
from engine.consistency_graph import ConsistencyGraph, GlobalConsistencyError

def test_global_equality_with_tol():
    g = ConsistencyGraph()
    g.add_claim("modA:p95", {"value": 100.0})
    g.add_claim("modB:p95", {"value": 105.0})
    g.relate_must_equal("modA:p95", "modB:p95", tol_pct=0.1)  # ±10%
    g.check()  # לא אמור לזרוק

def test_global_inconsistent_leq():
    g = ConsistencyGraph()
    g.add_claim("modA:err_rate", {"value": 0.12})
    g.add_claim("modB:slo_err_budget", {"value": 0.10})
    g.relate_leq("modA:err_rate", "modB:slo_err_budget")
    try:
        g.check()
        assert False, "should fail leq"
    except GlobalConsistencyError as e:
        assert "leq" in str(e)