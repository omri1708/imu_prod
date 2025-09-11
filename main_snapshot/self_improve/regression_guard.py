# imu_repo/self_improve/regression_guard.py
from __future__ import annotations
from typing import Dict, Any
from engine.config import snapshot, load_config, save_config
from metrics.aggregate import aggregate_metrics

def detect_regression(*, window_s: int=600, name: str="guarded_handler",
                      max_rel_p95_degrade: float=0.10, max_error_rate: float=0.02) -> Dict[str,Any]:
    """
    מזהה נסיגה בחלון הזמן האחרון:
      - p95 עלה ביותר מ-10% (ברירת מחדל)
      - או שיעור שגיאות > 2%
    מחזיר {"regressed": bool, "stats": {...}}
    """
    s = aggregate_metrics(name=name, bucket=None, window_s=window_s)
    lat = s.get("latency",{}) or {}
    p95 = float(lat.get("p95_ms") or 0.0)
    avg = float(lat.get("avg_ms") or 0.0)
    err = float(s.get("error_rate", 0.0))
    # השוואה לפשוט: אם avg קיים, נבדוק פער יחסי p95 לעומת avg כמדד дегראדציה (ללא בייסליין חיצוני)
    reg = False
    reasons=[]
    if avg>0.0 and p95 > (1.0 + max_rel_p95_degrade)*avg:
        reg=True; reasons.append("p95_relative_spike")
    if err > max_error_rate:
        reg=True; reasons.append("error_rate_spike")
    return {"regressed": reg, "reasons": reasons, "stats": s}

def rollback_with_snapshot(tag: str="regression")->str:
    """
    יוצר סנאפסוט (קונפיג+יומנים) ומחזיר נתיב. (מדיניות החזרה לקונפיג קודם — תוגדר ע"י caller)
    """
    path = snapshot(tag)
    return path