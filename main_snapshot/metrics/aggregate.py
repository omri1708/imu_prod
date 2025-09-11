# imu_repo/metrics/aggregate.py
from __future__ import annotations
import os, json, time, math
from typing import Any, Dict, Iterable, List, Tuple, Optional

LOG_ROOT = os.getenv("IMU_LOG_DIR", "/mnt/data/imu_repo/logs")
METRICS_F = os.path.join(LOG_ROOT, "metrics.jsonl")
ALERTS_F  = os.path.join(LOG_ROOT, "alerts.jsonl")

def _iter_jsonl(path: str) -> Iterable[Dict[str,Any]]:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def _in_window(ts_ms: int, now_ms: int, win_s: int) -> bool:
    return ts_ms >= now_ms - win_s*1000

def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals: return float("nan")
    if p<=0: return sorted_vals[0]
    if p>=100: return sorted_vals[-1]
    k = (len(sorted_vals)-1) * (p/100.0)
    f = math.floor(k); c = math.ceil(k)
    if f==c: return sorted_vals[int(k)]
    return sorted_vals[f] + (sorted_vals[c]-sorted_vals[f])*(k-f)

def aggregate_metrics(*,
                      name: str,
                      bucket: Optional[str]=None,
                      window_s: int=600) -> Dict[str,Any]:
    """
    מסכם מדדים עבור 'name' (למשל 'guarded_handler') בחלון זמן rolling.
    לוקט: p50/p95/p99/avg, ספירה, קצב/שניה, error_rate, evidence_gate_denied_rate.
    """
    now = int(time.time()*1000)
    vals: List[float] = []
    n_total = 0
    n_ok = 0
    # קרא metrics
    for m in _iter_jsonl(METRICS_F):
        ts = int(m.get("ts", 0))
        if not _in_window(ts, now, window_s): continue
        if m.get("name") != name: continue
        meta = m.get("meta", {}) or {}
        # תמיכה גם ב-top-level וגם בתוך meta
        b = meta.get("bucket") or "default"
        if bucket is not None and b != bucket: continue
        n_total += 1
        # תמיכה גם ב-latency_ms וגם ב-lat_ms
        lat = meta.get("latency_ms", meta.get("lat_ms"))
        if lat is not None:
            try: vals.append(float(lat))
            except Exception: pass
        if meta.get("ok") is True:
            n_ok += 1
    vals.sort()
    # קרא alerts — לשיעורי תקלות מסוגים שונים
    n_gate_denied = 0
    n_fail = 0
    for a in _iter_jsonl(ALERTS_F):
        ts = int(a.get("ts", 0))
        if not _in_window(ts, now, window_s): continue
        meta = a.get("meta", {}) or {}
        b = (a.get("bucket") or meta.get("bucket") or "default")
        if bucket is not None and b != bucket: continue
        ev = a.get("event")
        if ev == "evidence_gate_denied":
            n_gate_denied += 1
            n_fail += 1
        elif ev == "handler_failure":
            n_fail += 1

    error_rate = (n_fail / max(1, n_total)) if n_total else 0.0
    gate_denied_rate = (n_gate_denied / max(1, n_total)) if n_total else 0.0
    throughput_rps = (n_total / float(window_s)) if window_s>0 else float("nan")

    return {
        "name": name,
        "bucket": bucket or "all",
        "window_s": window_s,
        "count": n_total,
        "ok": n_ok,
        "error_rate": error_rate,
        "gate_denied_rate": gate_denied_rate,
        "throughput_rps": throughput_rps,
        "latency": {
            "avg_ms": (sum(vals)/len(vals)) if vals else float("nan"),
            "p50_ms": _percentile(vals, 50.0),
            "p95_ms": _percentile(vals, 95.0),
            "p99_ms": _percentile(vals, 99.0),
        },
    }