# imu_repo/engine/canary_autotune.py
from __future__ import annotations
from typing import Dict, Any

def suggest_next_percent(current_percent: int, headroom: float, policy: Dict[str,Any]) -> int:
    """
    headroom:
      latency:  limit/val  (גבוה=טוב)
      throughput: val/limit (גבוה=טוב)
      error_rate: limit/val (גבוה=טוב)
    policy["canary_autotune"] (בררת מחדל):
      {
        "accel_threshold": 1.3,  # מעל—אפשר להאיץ
        "decel_threshold": 1.0,  # מתחת—להאט/להקטין
        "accel_factor": 2.0,     # הכפלת אחוז
        "decel_factor": 0.5,     # חצי אחוז
        "min_step": 1,
        "max_step": 100
      }
    """
    cfg = (policy or {}).get("canary_autotune") or {}
    accel_thr = float(cfg.get("accel_threshold", 1.3))
    decel_thr = float(cfg.get("decel_threshold", 1.0))
    accel_f = float(cfg.get("accel_factor", 2.0))
    decel_f = float(cfg.get("decel_factor", 0.5))
    min_step = int(cfg.get("min_step", 1))
    max_step = int(cfg.get("max_step", 100))

    p = int(current_percent)
    if headroom >= accel_thr:
        np = int(max(p + min(max_step, max(min_step, round(p*(accel_f-1)))), p+min_step))
    elif headroom < decel_thr:
        np = int(max(min_step, round(p*decel_f)))
    else:
        # שמרני—קפיצה קטנה קדימה
        np = int(min(100, p + max(min_step, round(p*0.25))))
    return int(min(100, max(1, np)))