# engine/research/hypothesis_lab.py
from __future__ import annotations
import time, json, statistics
from typing import Any, Dict, List, Callable

class OfflineReport:
    def __init__(self):
        self.rows: List[Dict[str,Any]] = []
    def add(self, cfg: Dict[str,Any], metrics: Dict[str,Any]):
        self.rows.append({"cfg": cfg, "metrics": metrics})
    def summary(self) -> Dict[str,Any]:
        if not self.rows: return {"ok": False, "n": 0}
        p95s = [r["metrics"].get("p95_ms", 0.0) for r in self.rows]
        return {"ok": True, "n": len(self.rows), "p95_med": statistics.median(p95s)}


def run_offline_experiments(configs: List[Dict[str,Any]], *, runner: Callable[[Dict[str,Any]], Dict[str,Any]]) -> Dict[str,Any]:
    rep = OfflineReport()
    for cfg in configs:
        m = runner(cfg)
        rep.add(cfg, m)
    return {"ok": True, "report": rep.summary(), "rows": rep.rows}