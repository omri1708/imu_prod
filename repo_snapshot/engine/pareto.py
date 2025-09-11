# imu_repo/engine/pareto.py
from __future__ import annotations
from typing import List, Sequence

def pareto_front(points: Sequence[Sequence[float]]) -> List[int]:
    """
    קבלת חזית Pareto (מינימיזציה בכל הממדים).
    points[i] = [latency, cost, errors, distrust, energy, mem]
    מחזיר אינדקסים שאינם דומיננטיים.
    O(n^2) — מספיק טוב לכמות וריאציות קטנה.
    """
    n = len(points)
    if n == 0:
        return []
    dominated = [False]*n
    for i in range(n):
        if dominated[i]:
            continue
        Pi = points[i]
        for j in range(n):
            if i == j or dominated[i]:
                continue
            Pj = points[j]
            # j דומיננטי על i אם טוב/שווה בכל ממד וטוב לפחות בממד אחד
            better_or_eq_all = True
            better_at_least_one = False
            for a,b in zip(Pj, Pi):
                if a > b + 1e-12:   # גדול → גרוע (כי ממזערים)
                    better_or_eq_all = False
                    break
                if a < b - 1e-12:
                    better_at_least_one = True
            if better_or_eq_all and better_at_least_one:
                dominated[i] = True
    return [i for i in range(n) if not dominated[i]]