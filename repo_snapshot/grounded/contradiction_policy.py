# imu_repo/grounded/contradiction_policy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class MetricRule:
    # סטייה יחסית מותרת (לערכים נומריים), למשל 0.25 = 25%
    rel_tol: float = 0.25
    # סטייה מוחלטת מותרת (ms, יחידות וכו'), מתווספת ל-rel_tol
    abs_tol: float = 0.0
    # האם המדד "קריטי" (סתירה בו חוסמת Rollout)
    critical: bool = True

class ContradictionPolicy:
    """
    כללים בסיסיים למדדים ידועים. ניתן להרחיב בזמן ריצה.
    """
    def __init__(self):
        self.rules: Dict[str, MetricRule] = {
            "perf.p95_ms": MetricRule(rel_tol=0.50, abs_tol=150.0, critical=True),
            "ui.score":    MetricRule(rel_tol=0.20, abs_tol=5.0,   critical=False),
            "tests.passed":MetricRule(rel_tol=0.0,  abs_tol=0.0,   critical=True),
            "db.rows":     MetricRule(rel_tol=0.50, abs_tol=100.0, critical=False),
            # אפשר להוסיף בזמן ריצה:
            # self.rules["ext.metric"] = MetricRule(...)
        }
        # סף ציוני־עקביות (0–100). מתחת לסף → חוסם Rollout
        self.min_consistency_score: float = 80.0

    def set_rule(self, name: str, rule: MetricRule) -> None:
        self.rules[name] = rule

    def get_rule(self, name: str) -> MetricRule:
        return self.rules.get(name, MetricRule(rel_tol=0.30, abs_tol=0.0, critical=False))

    def set_min_score(self, score: float) -> None:
        self.min_consistency_score = float(score)

policy_singleton = ContradictionPolicy()