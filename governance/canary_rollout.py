# imu_repo/governance/canary_rollout.py
from __future__ import annotations
from typing import Dict, Any

class RolloutAction:
    PROMOTE = "promote"   # increase traffic share
    HOLD    = "hold"      # keep canary share
    ROLLBACK= "rollback"  # revert to baseline

class CanaryRollout:
    """
    Manage rollout based on observed KPIs during canary.
    thresholds example:
    {
      "max_error_rate": 0.01,
      "max_p95_latency_ms": 500.0,
      "promote_step": 0.2,  # increase traffic by 20% on success
    }
    """

    def __init__(self, thresholds: Dict[str, float]):
        self.th = thresholds

    def decide(self, canary_kpis: Dict[str,Any], traffic_share: float) -> Dict[str,Any]:
        # canary_kpis contains keys from obs.kpi.summarize_runs
        err = canary_kpis.get("error_rate", 0.0)
        p95 = canary_kpis.get("p95", 0.0)
        if err > self.th.get("max_error_rate", 1.0) or p95 > self.th.get("max_p95_latency_ms", float("inf")):
            return {"action": RolloutAction.ROLLBACK, "reason": "kpi_violation", "traffic_share": traffic_share}

        step = self.th.get("promote_step", 0.1)
        new_share = min(1.0, max(traffic_share, traffic_share + step))
        if new_share > traffic_share:
            return {"action": RolloutAction.PROMOTE, "reason": "healthy", "traffic_share": new_share}
        return {"action": RolloutAction.HOLD, "reason": "at_max", "traffic_share": traffic_share}
