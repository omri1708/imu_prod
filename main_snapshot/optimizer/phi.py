# imu_repo/optimizer/phi.py
from __future__ import annotations
from typing import Dict, Any, List
import math

def _nz(x: float, eps: float = 1e-9) -> float:
    return x if x > eps else eps

def compute_phi(metrics: Dict[str, Any]) -> float:
    """
    Φ = משלב ממדד שגיאות, p95_latency, וצריכת משאבים לכדי ציון סקלרי.
    נמוך יותר => עדיף. כל רכיב מנורמל ביחס לסקלה סבירה.
    metrics לדוגמה:
    {
      "latency_ms": 42.0,   # זמן לריצה אחת
      "p95": 120.0,         # אם זמין
      "cpu_steps": 2.0e5,
      "mem_kb": 16384,
      "io_calls": 12,
      "error": False
    }
    """
    err = 1.0 if metrics.get("error", False) else 0.0
    p95 = float(metrics.get("p95", metrics.get("latency_ms", 0.0)))
    cpu = float(metrics.get("cpu_steps", 0.0))
    mem = float(metrics.get("mem_kb", 0.0))
    io  = float(metrics.get("io_calls", 0.0))

    # נרמול לסקאלות שמרניות (ניתן לכייל במציאות):
    norm_p95 = p95 / 500.0          # 500ms כ-benchmark
    norm_cpu = cpu / 5.0e5          # 500k steps
    norm_mem = mem / 65536.0        # 64MB
    norm_io  = io  / 1000.0

    # משקולות (ניתן לכייל/ללמוד):
    w_err, w_p95, w_cpu, w_mem, w_io = 10.0, 3.0, 1.0, 0.5, 0.3

    phi = (w_err * err) + (w_p95 * norm_p95) + (w_cpu * norm_cpu) + (w_mem * norm_mem) + (w_io * norm_io)
    return float(phi)

def suite_phi(runs: List[Dict[str, Any]]) -> float:
    """
    מקבל רשימת ריצות בפורמט:
    [{"kind":"ok"/"error","metrics":{...}}, ...]
    מחזיר Φ כולל (ממוצע גאומטרי כדי להעניש זנבות)
    """
    vals: List[float] = []
    for r in runs:
        m = dict(r.get("metrics", {}))
        m["error"] = (r.get("kind") == "error") or bool(m.get("error", False))
        vals.append(max(compute_phi(m), 1e-9))
    if not vals:
        return float("inf")
    # ממוצע גאומטרי
    log_avg = sum(math.log(v) for v in vals) / len(vals)
    return float(math.exp(log_avg))
