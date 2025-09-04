# imu_repo/obs/kpi.py
from __future__ import annotations
from typing import Dict, Any, List
import math, statistics, time
import os, json, time, statistics

class KPIRecorder:
    """Record metrics for runs and compute aggregates (p50/p95/p99, error-rate)."""

    def __init__(self):
        self.events: List[Dict[str,Any]] = []

    def record(self, kind:str, metrics:Dict[str,Any]):
        ev = {"ts": time.time(), "kind": kind, "metrics": metrics}
        self.events.append(ev)

    @staticmethod
    def _percentile(values: List[float], p: float) -> float:
        if not values: return float("nan")
        values = sorted(values)
        k = (len(values)-1) * p
        f = math.floor(k); c = math.ceil(k)
        if f == c: return values[int(k)]
        d0 = values[f] * (c-k)
        d1 = values[c] * (k-f)
        return d0 + d1

    def summarize(self, key: str) -> Dict[str, float]:
        vals = [float(ev["metrics"].get(key, 0.0)) for ev in self.events if key in ev["metrics"]]
        return {
            "count": float(len(vals)),
            "avg": statistics.fmean(vals) if vals else float("nan"),
            "p50": self._percentile(vals, 0.50),
            "p95": self._percentile(vals, 0.95),
            "p99": self._percentile(vals, 0.99)
        }

    def error_rate(self) -> float:
        total = len(self.events)
        if total == 0: return 0.0
        errors = sum(1 for ev in self.events if ev["kind"] in ("error", "contract_violation", "vm_error"))
        return errors / total

def summarize_runs(runs: List[Dict[str,Any]], latency_key="latency_ms") -> Dict[str, Any]:
    """
    runs: [{"metrics": {...}, "kind": "ok/error"}]
    """
    rec = KPIRecorder()
    for r in runs:
        rec.record(r.get("kind","ok"), r.get("metrics", {}))
    out = rec.summarize(latency_key)
    out["error_rate"] = rec.error_rate()
    return out


class KPI:
    def __init__(self, path: str = ".imu_state/kpi.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path,"w",encoding="utf-8"): pass

    def record(self, latency_ms: float, error: bool = False):
        rec = {"ts": time.time(), "latency_ms": float(latency_ms), "error": bool(error)}
        with open(self.path,"a",encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def _load(self, limit: int = 1000) -> List[Dict[str,Any]]:
        out=[]
        with open(self.path,"r",encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try: out.append(json.loads(line))
                except Exception: pass
        return out[-limit:]

    def snapshot(self, limit: int = 1000) -> Dict[str,Any]:
        data=self._load(limit=limit)
        if not data:
            return {"count":0,"p95":0.0,"error_rate":0.0,"avg":0.0}
        lats=[d["latency_ms"] for d in data]
        lats_sorted=sorted(lats)
        p95=lats_sorted[int(max(0,len(lats_sorted)*0.95)-1)]
        err=sum(1 for d in data if d["error"])
        return {
            "count": len(data),
            "avg": sum(lats)/len(lats),
            "p95": p95,
            "error_rate": err/max(1,len(data))
        }