# -*- coding: utf-8 -*-
"""
הרצה קצרה שמדגימה למידה:
- "מתבוננת" בכמה resource_required
- מעדכנת policy.yaml ומייצרת adapter אוטומטי לכלי שחסר שוב ושוב
"""
from __future__ import annotations
from learning.policy_learner import PolicyTuner, MetricsSnapshot

def main():
    tuner = PolicyTuner("./executor/policy.yaml","./adapters/generated")
    tuner.load()
    # דמה: נניח שקיבלנו 3 פעמים חוסר בכלי 'curl'
    for _ in range(3):
        tuner.observe_resource_required("tool:curl")
    res = tuner.tune_from_metrics(MetricsSnapshot(total_commits=10))
    print("tune result:", res)

if __name__=="__main__":
    main()
