# imu_repo/tests/test_stage67_self_sustaining.py
from __future__ import annotations
import os, json, time, shutil
from typing import Any, Dict, List
from alerts.notifier import ROOT as LOG_ROOT
from metrics.aggregate import aggregate_metrics
from sla.policy import SlaSpec, evaluate
from safe_progress.auto_rollout import decide, DEC_PROMOTE, DEC_HOLD, DEC_ROLLBACK
from self_improve.planner import plan_from_stats
from self_improve.patcher import apply_all
from self_improve.ab_runner import run_bucket, simple_workload
from self_improve.regression_guard import detect_regression, rollback_with_snapshot
from engine.config import load_config, save_config

def _reset_logs():
    os.makedirs(LOG_ROOT, exist_ok=True)
    for fn in ("metrics.jsonl","alerts.jsonl"):
        p = os.path.join(LOG_ROOT, fn)
        if os.path.exists(p): os.remove(p)

def _print(msg: str):
    print(msg, flush=True)

def run():
    # 1) אפס סביבה וכתוב קונפיג ברירת מחדל
    _reset_logs()
    cfg = load_config()
    cfg["ws"]["chunk_size"] = 64000
    cfg["ws"]["max_pending_msgs"] = 1024
    cfg["ws"]["permessage_deflate"] = True
    cfg["guard"]["min_trust"] = 0.7
    cfg["guard"]["max_age_s"] = 3600
    save_config(cfg)

    # 2) דמה מצב בעייתי (baseline): p95 גבוה (80ms), ושיעור gate_denied נמוך — נרצה להוריד p95
    run_bucket("baseline", simple_workload, {"n": 250, "lat_ms": 80.0, "metric_name":"guarded_handler"})
    base = aggregate_metrics(name="guarded_handler", bucket="baseline", window_s=3600)
    _print(f"baseline p95: {base['latency']['p95_ms']}")

    # 3) גזור FixPlan מתוך baseline
    plans = plan_from_stats(base)
    assert any(p.reason=="p95_high" for p in plans), "expected p95_high plan"
    cfg2 = apply_all(plans)
    _print("applied plan(s): " + json.dumps([p.as_dict() for p in plans], ensure_ascii=False))

    # 4) הפעל canary עם שיפור מדומה (60ms) כדי לאפשר החלטה — מדמה שהשינויים עזרו
    run_bucket("canary", simple_workload, {"n": 250, "lat_ms": 60.0, "metric_name":"guarded_handler"})
    can = aggregate_metrics(name="guarded_handler", bucket="canary", window_s=3600)
    _print(f"canary p95: {can['latency']['p95_ms']}")

    # 5) בדיקת SLA קשיח
    spec = SlaSpec("default", p95_ms=75.0, max_error_rate=0.02, max_gate_denied_rate=0.02, min_throughput_rps=0.0)
    sla_can = evaluate(can, spec)
    assert sla_can["ok"] is True, "canary should pass SLA"

    # 6) החלטת rollout — נדרוש לפחות not_worse; כאן יש גם שיפור
    d = decide(window_s=3600, name="guarded_handler", sla=spec, require_improvement=False)
    _print("rollout decision: " + d["decision"])
    assert d["decision"] in (DEC_PROMOTE, DEC_HOLD), "should not rollback"

    # 7) “פרוס” (לצורך הטסט פשוט נאפס דגל bucket ונמשיך לייצר תנועה כללית)
    os.environ["IMU_BUCKET"] = "default"
    _reset_logs()
    # אחרי פריסה — דמה דגרדציה כדי לבדוק regression stop
    # קודם כמה קריאות טובות:
    run_bucket("default", simple_workload, {"n": 50, "lat_ms": 60.0, "metric_name":"guarded_handler"})
    # ואז זינוק בזנב:
    run_bucket("default", simple_workload, {"n": 50, "lat_ms": 120.0, "metric_name":"guarded_handler", "fail_every": 10})

    reg = detect_regression(window_s=3600, name="guarded_handler", max_rel_p95_degrade=0.10, max_error_rate=0.05)
    if reg["regressed"]:
        snap = rollback_with_snapshot(tag="regression")
        _print("regression detected; snapshot at: " + snap)
    else:
        _print("no regression detected")

    print("OK")
    return 0

if __name__=="__main__":
    raise SystemExit(run())