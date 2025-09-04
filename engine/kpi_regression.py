# imu_repo/engine/kpi_regression.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple
import json, math, os

class KPIRegressionBlocked(Exception): ...
class KPIDataError(Exception): ...

@dataclass
class KPISummary:
    n: int
    ok: int
    error: int
    schema_errors: int
    p95_ms: float
    mean_ms: float
    error_rate: float
    schema_error_rate: float

def _quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    q = min(max(q, 0.0), 1.0)
    idx = q * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_vals[lo])
    h = idx - lo
    return float(sorted_vals[lo] * (1.0 - h) + sorted_vals[hi] * h)

def summarize(records: Iterable[Dict[str, Any]]) -> KPISummary:
    lat: List[float] = []
    ok = 0
    err = 0
    schema_err = 0
    n = 0
    for r in records:
        n += 1
        # ok: bool; latency_ms: number; schema_errors: int (אופציונלי)
        ok_flag = bool(r.get("ok", False))
        if ok_flag:
            ok += 1
        else:
            err += 1
        se = int(r.get("schema_errors", 0) or 0)
        schema_err += se
        lm = r.get("latency_ms")
        if lm is not None:
            try:
                lat.append(float(lm))
            except Exception:
                # נתעלם מערכים לא ניתנים להמרה
                pass
    lat.sort()
    p95 = _quantile(lat, 0.95) if lat else float("nan")
    mean = (sum(lat)/len(lat)) if lat else float("nan")
    erate = (err / n) if n else 0.0
    serate = (schema_err / n) if n else 0.0
    return KPISummary(n=n, ok=ok, error=err, schema_errors=schema_err,
                      p95_ms=p95, mean_ms=mean,
                      error_rate=erate, schema_error_rate=serate)

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise KPIDataError(f"missing KPI file: {path}")
    out: List[Dict[str,Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: 
                continue
            try:
                out.append(json.loads(ln))
            except Exception as e:
                raise KPIDataError(f"invalid json line in {path}: {e}")
    return out

def load_and_summarize(path: str) -> KPISummary:
    return summarize(_read_jsonl(path))

def compare_kpis(
    base: KPISummary,
    cand: KPISummary,
    policy: Dict[str,Any]
) -> Dict[str,Any]:
    """
    Gate ברירות־מחדל שמרניות:
      - p95(candidate) <= p95(baseline) + 50ms
      - error_rate(candidate) <= error_rate(baseline) + 1%
      - אם block_on_schema_regression=True (ברירת־מחדל): schema_error_rate(candidate) <= baseline
    """
    max_p95_inc = float(policy.get("max_p95_increase_ms", 50.0))
    max_err_inc = float(policy.get("max_error_rate_increase", 0.01))
    block_schema = bool(policy.get("block_on_schema_regression", True))
    # NaN התנהגות: אם אין לטנסיות — לא נאכוף p95
    p95_ok = True
    p95_delta = float("nan")
    if not math.isnan(base.p95_ms) and not math.isnan(cand.p95_ms):
        p95_delta = cand.p95_ms - base.p95_ms
        p95_ok = (p95_delta <= max_p95_inc)
    err_delta = cand.error_rate - base.error_rate
    err_ok = (err_delta <= max_err_inc)
    schema_ok = True
    schema_delta = cand.schema_error_rate - base.schema_error_rate
    if block_schema:
        schema_ok = (schema_delta <= 0.0)

    verdict = p95_ok and err_ok and schema_ok
    result = {
        "baseline": base.__dict__,
        "candidate": cand.__dict__,
        "deltas": {
            "p95_ms": p95_delta,
            "error_rate": err_delta,
            "schema_error_rate": schema_delta
        },
        "thresholds": {
            "max_p95_increase_ms": max_p95_inc,
            "max_error_rate_increase": max_err_inc,
            "block_on_schema_regression": block_schema
        },
        "ok": verdict
    }
    if not verdict:
        reasons = []
        if not p95_ok:
            reasons.append(f"p95 regression {p95_delta:.2f}ms > {max_p95_inc}ms")
        if not err_ok:
            reasons.append(f"error-rate regression {err_delta:.4f} > {max_err_inc}")
        if not schema_ok:
            reasons.append(f"schema-error-rate regression {schema_delta:.4f} > 0")
        raise KPIRegressionBlocked("; ".join(reasons))
    return result

def gate_from_files(baseline_path: str, candidate_path: str, policy: Dict[str,Any]) -> Dict[str,Any]:
    base = load_and_summarize(baseline_path)
    cand = load_and_summarize(candidate_path)
    return compare_kpis(base, cand, policy)