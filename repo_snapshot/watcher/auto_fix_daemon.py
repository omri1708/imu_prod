# imu_repo/watcher/auto_fix_daemon.py
from __future__ import annotations
import os, shutil, json, time
from typing import Callable, Dict, Any, Optional, Tuple, List

from sla.policy import SlaSpec, evaluate
from metrics.aggregate import aggregate_metrics
from self_improve.planner import plan_from_stats
from self_improve.apply import apply_with_executors
from self_improve.ab_runner import run_bucket, Workload
from safe_progress.auto_rollout import decide, DEC_PROMOTE, DEC_HOLD, DEC_ROLLBACK
from self_improve.regression_guard import detect_regression, rollback_with_snapshot
from engine.config import load_config, save_config, snapshot, CFG_FILE

ROOT = "/mnt/data/imu_repo"
STATE_DIR = os.path.join(ROOT, "state")
os.makedirs(STATE_DIR, exist_ok=True)
STATE_FILE = os.path.join(STATE_DIR, "rollout_state.json")

def _write_state(obj: Dict[str,Any]) -> None:
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2))
    os.replace(tmp, STATE_FILE)

def _read_state() -> Dict[str,Any]:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        return json.load(open(STATE_FILE, "r", encoding="utf-8"))
    except Exception:
        return {}

def _restore_config_from_snapshot(snap_path: str) -> bool:
    """
    משחזר runtime.json מסנאפשוט (אם קיים), ומחזיר True אם שוחזר.
    """
    src = os.path.join(snap_path, "runtime.json")
    if os.path.exists(src):
        shutil.copy2(src, CFG_FILE)
        return True
    return False

def run_once(*,
             name: str = "guarded_handler",
             window_s: int = 600,
             sla: Optional[SlaSpec] = None,
             workload: Optional[Workload] = None,
             baseline_params: Optional[Dict[str,Any]] = None,
             canary_params: Optional[Dict[str,Any]] = None,
             require_improvement: bool = False,
             min_rel_impr: float = 0.05,
             seed_if_empty: bool = True) -> Dict[str,Any]:
    """
    מריץ מחזור Auto-Fix יחיד:
      1) קורא סטטיסטיקות baseline; אם אין תנועה וביקשת seed_if_empty — מייצר baseline via workload.
      2) אם baseline עובר SLA — אין מה לתקן.
      3) אחרת: מפיק FixPlan(s), יישום Executors, A/B (baseline/canary), החלטה promote/hold/rollback.
      4) במקרה rollback — יוצר snapshot ומנסה לשחזר קונפיג קודם.
    """
    # 1) baseline
    base = aggregate_metrics(name=name, bucket="baseline", window_s=window_s)
    if base.get("count", 0) == 0 and seed_if_empty:
        if workload and baseline_params is not None:
            run_bucket("baseline", workload, baseline_params)
            base = aggregate_metrics(name=name, bucket="baseline", window_s=window_s)

    # אם יש SLA, בדוק
    eval_res = None
    if sla is not None and base.get("count", 0) > 0:
        eval_res = evaluate(base, sla)
        if eval_res["ok"]:
            state = {"status":"baseline_ok", "baseline": base, "sla_eval": eval_res}
            _write_state(state)
            return state

    # 2) הפקת FixPlan(s)
    plans = plan_from_stats(base if base.get("count",0)>0 else {"latency":{"p95_ms": 1e9}})
    # Snapshot לפני שינוי קונפיג
    snap_pre = snapshot("pre_candidate")
    applied_summary = apply_with_executors(plans)

    # 3) A/B
    if workload is not None:
        if baseline_params is not None:
            run_bucket("baseline", workload, baseline_params)
        if canary_params is not None:
            run_bucket("canary", workload, canary_params)

    # 4) החלטת רולאאוט
    dec = decide(window_s=window_s, name=name, sla=sla, require_improvement=require_improvement, min_rel_impr=min_rel_impr)

    # 5) טיפול בהחלטה
    final = {
        "decision": dec["decision"],
        "baseline": dec["baseline"],
        "canary": dec["canary"],
        "sla": dec.get("sla"),
        "comparison": dec.get("comparison"),
        "plans": [p.as_dict() for p in getattr(plans, "__iter__", lambda:[])()],
        "applied": applied_summary,
        "snap_pre_candidate": snap_pre
    }

    if dec["decision"] == DEC_PROMOTE:
        # אחרי "פריסה": ננטר נסיגה כללית (bucket=all) ונשמור סנאפשוט.
        reg = detect_regression(window_s=window_s, name=name)
        final["post_regression_check"] = reg
        if reg["regressed"]:
            snap = rollback_with_snapshot(tag="regression_after_promote")
            _restore_config_from_snapshot(snap_pre)
            final["rollback_snapshot"] = snap
            final["rolled_back"] = True
        else:
            final["rolled_back"] = False

    elif dec["decision"] == DEC_ROLLBACK:
        snap = rollback_with_snapshot(tag="rollout_denied")
        # שחזור ישיר לקונפיג הקודם
        _restore_config_from_snapshot(snap_pre)
        final["rollback_snapshot"] = snap
        final["rolled_back"] = True

    else:  # HOLD
        final["rolled_back"] = False

    _write_state(final)
    return final

def run_forever(*,
                name: str = "guarded_handler",
                window_s: int = 600,
                period_s: int = 30,
                sla: Optional[SlaSpec] = None,
                workload: Optional[Workload] = None,
                baseline_params: Optional[Dict[str,Any]] = None,
                canary_params: Optional[Dict[str,Any]] = None,
                require_improvement: bool = False,
                min_rel_impr: float = 0.05,
                cycles: Optional[int] = None) -> None:
    """
    לולאת שיפור מתמשכת. אם cycles=None – רצה לנצח.
    """
    i = 0
    while True:
        run_once(name=name, window_s=window_s, sla=sla, workload=workload,
                 baseline_params=baseline_params, canary_params=canary_params,
                 require_improvement=require_improvement, min_rel_impr=min_rel_impr)
        i += 1
        if cycles is not None and i >= cycles:
            break
        time.sleep(period_s)