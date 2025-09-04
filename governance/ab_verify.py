# imu_repo/governance/ab_verify.py
from __future__ import annotations
from typing import Dict, Any
from obs.kpi import summarize_runs

class ABDecision:
    def __init__(self, passed: bool, report: Dict[str,Any]):
        self.passed = passed
        self.report = report

class ABVerifier:
    """
    Compare baseline vs candidate runs using KPI thresholds.
    thresholds example:
    {
      "max_error_rate": 0.01,
      "max_p95_latency_ms": 500.0,
      "max_regression_p95_ms": 5.0,  # candidate p95 - baseline p95 must be <= this
    }
    """

    def __init__(self, thresholds: Dict[str, float]):
        self.th = thresholds

    def compare(self, baseline_runs: list[dict], candidate_runs: list[dict]) -> ABDecision:
        base = summarize_runs(baseline_runs)
        cand = summarize_runs(candidate_runs)
        report = {"baseline": base, "candidate": cand, "thresholds": self.th}

        # absolute gates for candidate
        if cand["error_rate"] > self.th.get("max_error_rate", 1.0):
            return ABDecision(False, {**report, "reason": "error_rate_exceeded"})
        if cand["p95"] > self.th.get("max_p95_latency_ms", float("inf")):
            return ABDecision(False, {**report, "reason": "p95_latency_exceeded"})

        # relative regression gate vs baseline
        reg_p95 = cand["p95"] - base["p95"]
        if reg_p95 > self.th.get("max_regression_p95_ms", float("inf")):
            return ABDecision(False, {**report, "reason": "p95_regression", "delta_ms": reg_p95})

        return ABDecision(True, {**report, "reason": "ok"})
