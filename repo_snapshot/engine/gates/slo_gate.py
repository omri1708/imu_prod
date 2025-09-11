# imu_repo/engine/gates/slo_gate.py
from __future__ import annotations
from typing import Dict, Any
from runtime.metrics import metrics

class SLOGate:
    """
    בודק SLO גלובלי של ה-mesh:
      cfg = {
        "p95_ms": {"mesh.router.request": 800},
        "error_rate_max": 0.05,      # שיעור שגיאות מותר
        "min_requests": 10           # דרישת נפח לפני שיפסל
      }
    """
    def __init__(self, p95_ms: Dict[str,float] | None=None,
                 error_rate_max: float=0.05,
                 min_requests: int=10):
        self.p95_ms = p95_ms or {}
        self.error_rate_max = float(error_rate_max)
        self.min_requests = int(min_requests)

    def check(self) -> Dict[str,Any]:
        snap = metrics.snapshot()
        lat = snap["latencies"]; ctr = snap["counters"]
        viol = []

        # p95
        for k, lim in self.p95_ms.items():
            arr = lat.get(k, [])
            if not arr: 
                continue
            arr2 = sorted(arr)
            p95 = arr2[int(0.95*(len(arr2)-1))]
            if p95 > float(lim):
                viol.append(("p95", k, p95, lim))

        # error rate
        total = int(ctr.get("mesh.router.total", 0))
        errors = int(ctr.get("mesh.router.errors", 0))
        erate = (errors/total) if total>0 else 0.0
        er_ok = True
        if total >= self.min_requests:
            er_ok = erate <= self.error_rate_max
            if not er_ok:
                viol.append(("error_rate", erate, self.error_rate_max, total))

        return {"ok": len(viol)==0, "violations": viol, "total": total, "errors": errors, "error_rate": erate}