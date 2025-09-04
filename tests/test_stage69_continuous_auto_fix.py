# imu_repo/tests/test_stage69_continuous_auto_fix.py
from __future__ import annotations
import os, json
from typing import Dict, Any
from alerts.notifier import ROOT as LOG_ROOT
from sla.policy import SlaSpec
from watcher.auto_fix_daemon import run_once, _read_state
from self_improve.ab_runner import simple_workload
from engine.config import load_config, save_config

def _reset_logs():
    os.makedirs(LOG_ROOT, exist_ok=True)
    for fn in ("metrics.jsonl","alerts.jsonl"):
        p = os.path.join(LOG_ROOT, fn)
        if os.path.exists(p): os.remove(p)

def run():
    # אפס יומנים וקבע קונפיג ברירת מחדל
    _reset_logs()
    cfg = load_config()
    cfg["ws"]["chunk_size"] = 64000
    cfg["ws"]["max_pending_msgs"] = 1024
    cfg["ws"]["permessage_deflate"] = True
    cfg["guard"]["min_trust"] = 0.7
    cfg["guard"]["max_age_s"] = 3600
    save_config(cfg)

    # SLA קשיח: p95<=90ms, error_rate<=2%, gate_denied<=2%
    spec = SlaSpec("strict", p95_ms=90.0, max_error_rate=0.02, max_gate_denied_rate=0.02, min_throughput_rps=0.0)

    # Baseline שמפר SLA (p95=120ms)
    base_params = {"n": 250, "lat_ms": 120.0, "metric_name":"guarded_handler"}
    # Canary “משופר” (p95=70ms) — מדמה שה‐FixPlan ו־Executors שיפרו את המערכת
    can_params  = {"n": 250, "lat_ms": 70.0, "metric_name":"guarded_handler"}

    out = run_once(name="guarded_handler",
                   window_s=3600,
                   sla=spec,
                   workload=simple_workload,
                   baseline_params=base_params,
                   canary_params=can_params,
                   require_improvement=False,
                   min_rel_impr=0.05,
                   seed_if_empty=True)

    assert out["decision"] in ("promote","hold","rollback")
    st = _read_state()
    assert st.get("decision") == out["decision"]
    print("decision:", out["decision"])
    print("OK")
    return 0

if __name__=="__main__":
    raise SystemExit(run())