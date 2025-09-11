# imu_repo/engine/gates/runtime_budget.py
from __future__ import annotations
from typing import Dict, Any
from runtime.metrics import metrics

class RuntimeBudgetGate:
    """
    בודק תקציב־ריצה: פ95 לזמנים וקאונטרים (למשל TPS בפועל).
      config = {
         "p95": {"sandbox.http_get": 800, "sandbox.sleep_ms": 600},
         "counters_max": {"sandbox.http_get.count": 5}
      }
    """
    def __init__(self, p95: Dict[str,float] | None=None, counters_max: Dict[str,int] | None=None):
        self.p95_limits = p95 or {}
        self.counter_limits = counters_max or {}

    def check(self) -> Dict[str,Any]:
        snap = metrics.snapshot()
        lat = snap["latencies"]; ctr = snap["counters"]
        bad=[]
        for k, lim in self.p95_limits.items():
            # חישוב p95 מתוך הסנאפשוט (ללא side-effect)
            arr = lat.get(k, [])
            ok = True
            if arr:
                arr2 = sorted(arr)
                idx = int(0.95*(len(arr2)-1))
                p95 = arr2[idx]
                if p95 > float(lim):
                    ok = False
                val = p95
            else:
                val = None  # אין נתון → לא נכשל (למעט אם תרצה להפוך לחובה)
            if not ok:
                bad.append(("p95", k, val, lim))
        for k, lim in self.counter_limits.items():
            v = int(ctr.get(k, 0))
            if v > int(lim):
                bad.append(("counter", k, v, lim))
        return {"ok": len(bad)==0, "violations": bad}