# imu_repo/sla/policy.py
from __future__ import annotations
from typing import Dict, Any, Optional

class SlaSpec:
    __slots__ = ("name","p95_ms","max_error_rate","max_gate_denied_rate","min_throughput_rps")
    def __init__(self, name: str, *, p95_ms: float, max_error_rate: float, max_gate_denied_rate: float, min_throughput_rps: float=0.0):
        self.name=name; self.p95_ms=p95_ms
        self.max_error_rate=max_error_rate
        self.max_gate_denied_rate=max_gate_denied_rate
        self.min_throughput_rps=min_throughput_rps

def evaluate(stats: Dict[str,Any], spec: SlaSpec) -> Dict[str,Any]:
    lat = stats.get("latency", {})
    p95 = float(lat.get("p95_ms") or float("inf"))
    ok_p95 = p95 <= spec.p95_ms
    err = float(stats.get("error_rate", 0.0))
    ok_err = err <= spec.max_error_rate
    gate = float(stats.get("gate_denied_rate", 0.0))
    ok_gate = gate <= spec.max_gate_denied_rate
    thr = float(stats.get("throughput_rps", 0.0))
    ok_thr = thr >= spec.min_throughput_rps

    ok_all = all([ok_p95, ok_err, ok_gate, ok_thr])
    return {
        "ok": ok_all,
        "checks": {
            "p95_ms": {"ok": ok_p95, "actual": p95, "limit": spec.p95_ms},
            "error_rate": {"ok": ok_err, "actual": err, "limit": spec.max_error_rate},
            "gate_denied_rate": {"ok": ok_gate, "actual": gate, "limit": spec.max_gate_denied_rate},
            "throughput_rps": {"ok": ok_thr, "actual": thr, "limit": spec.min_throughput_rps, "type":"min"},
        }
    }

def compare(baseline: Dict[str,Any], canary: Dict[str,Any], *, require_improvement: bool=False, min_rel_impr: float=0.05) -> Dict[str,Any]:
    """
    השוואה בין baseline ל-canary:
      - אם require_improvement: דרוש שיפור יחסי ב-p95 של לפחות min_rel_impr (5% כברירת מחדל).
      - אחרת: דרוש ש-canary לא נחות (p95 לא גדול יותר, ושיעורי כשל לא גבוהים).
    """
    b = float(baseline.get("latency",{}).get("p95_ms") or float("inf"))
    c = float(canary.get("latency",{}).get("p95_ms") or float("inf"))
    berr = float(baseline.get("error_rate",0.0)); cerr = float(canary.get("error_rate",0.0))
    bg = float(baseline.get("gate_denied_rate",0.0)); cg = float(canary.get("gate_denied_rate",0.0))
    # קריטריונים
    not_worse = (c <= b) and (cerr <= berr) and (cg <= bg)
    improved = (b - c) / max(1.0, b) >= float(min_rel_impr)
    ok = (not_worse if not require_improvement else improved)
    return {
        "ok": ok,
        "baseline_p95": b,
        "canary_p95": c,
        "not_worse": not_worse,
        "improved": improved
    }