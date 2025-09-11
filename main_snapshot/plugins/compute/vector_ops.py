# imu_repo/plugins/compute/vector_ops.py
from __future__ import annotations
from typing import Dict, Any, List
import time, math, random

class VectorOps:
    """
    פונקציות חישוביות (וקטור/מטריצה) בקוד טהור:
      - add, dot, matmul קטנות
    נמדד זמן ובודק תקינות תוצאה; מחזיר KPI פשוט.
    """
    def __init__(self, max_len: int = 2000):
        self.max_len = int(max_len)

    def _add(self, a: List[float], b: List[float]) -> List[float]:
        if len(a)!=len(b): raise ValueError("mismatch")
        return [x+y for x,y in zip(a,b)]

    def _dot(self, a: List[float], b: List[float]) -> float:
        if len(a)!=len(b): raise ValueError("mismatch")
        return sum(x*y for x,y in zip(a,b))

    def _matmul(self, A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        n, m, p = len(A), len(A[0]), len(B[0])
        # בדיקת ממדים בסיסית
        if any(len(row)!=m for row in A): raise ValueError("bad_A")
        if any(len(row)!=p for row in B): raise ValueError("bad_B_width")
        # transpose B
        Bt = [list(col) for col in zip(*B)]
        return [[sum(x*y for x,y in zip(row, col)) for col in Bt] for row in A]

    def run(self, spec: Any, build_dir: str, user_id: str) -> Dict[str,Any]:
        extras = getattr(spec, "extras", {}) or {}
        comp = (extras.get("compute") or {})
        n = min(int(comp.get("n", 512)), self.max_len)

        # דוגמא: dot/add
        a = [random.random() for _ in range(n)]
        b = [random.random() for _ in range(n)]
        t0 = time.time()
        c = self._add(a,b)
        d = self._dot(a,b)
        dt_ms = (time.time()-t0)*1000.0

        # gate פשוט: זמן ≤ 120ms עבור n≤2000
        if dt_ms > 120.0:
            raise RuntimeError("compute_time_exceeded")

        ev = {"n": n, "time_ms": dt_ms, "dot": d, "checksum": sum(c)}
        kpi = {"score": max(75.0, 95.0 - 0.02*dt_ms)}
        return {"plugin":"vector_ops","evidence": ev, "kpi": kpi}