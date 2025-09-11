# imu_repo/tests/test_stage68_auto_fix.py
from __future__ import annotations
import importlib, sys, os, json
from typing import Any, Dict, List
from alerts.notifier import ROOT as LOG_ROOT
from metrics.aggregate import aggregate_metrics
from self_improve.planner import plan_from_stats
from self_improve.apply import apply_with_executors
from engine.config import load_config, save_config

def _reset_logs():
    os.makedirs(LOG_ROOT, exist_ok=True)
    for fn in ("metrics.jsonl","alerts.jsonl"):
        p = os.path.join(LOG_ROOT, fn)
        if os.path.exists(p): os.remove(p)

def _fake_stats()->Dict[str,Any]:
    # מייצר סטטיסטיקות שמחייבות שיפור p95 וגם throughput
    return {
        "name": "guarded_handler",
        "bucket": "baseline",
        "window_s": 600,
        "count": 500,
        "ok": 490,
        "error_rate": 0.01,
        "gate_denied_rate": 0.005,
        "throughput_rps": 0.2,
        "latency": {
            "avg_ms": 70.0,
            "p50_ms": 60.0,
            "p95_ms": 120.0,
            "p99_ms": 150.0
        }
    }

def run():
    _reset_logs()
    # קונפיג ראשוני
    cfg = load_config()
    cfg.setdefault("ws", {}).update({"chunk_size": 64000, "max_pending_msgs": 1024, "permessage_deflate": True})
    cfg.setdefault("guard", {}).update({"min_trust": 0.7, "max_age_s": 3600})
    cfg.setdefault("db", {}).update({"sandbox": True, "max_conn": 8, "encrypt_at_rest": True})
    save_config(cfg)

    # 1) הפק תכניות תיקון מהסטטיסטיקות
    stats = _fake_stats()
    plans = plan_from_stats(stats)
    assert plans, "expected auto FixPlan(s)"

    # 2) החל עם מפעילים + צור בדיקות
    summary = apply_with_executors(plans)
    tests = summary.get("tests", [])
    assert tests, "expected generated tests"
    # 3) הרץ את כל הבדיקות שנוצרו
    root = "/mnt/data/imu_repo"
    if root not in sys.path: sys.path.append(root)
    for t in tests:
        mod = importlib.import_module(t["module"])
        ok = bool(getattr(mod, "run")())
        assert ok, f"test failed: {t['module']}"

    print("OK")
    return 0

if __name__=="__main__":
    raise SystemExit(run())