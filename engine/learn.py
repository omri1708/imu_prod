# imu_repo/engine/learn.py
from __future__ import annotations
from typing import Dict, Any, Optional, List
import time

from grounded.claims import current
from engine.learn_store import _task_key, append_history, load_history, load_baseline, save_baseline
from engine.convergence import has_converged, regression_guard

PROMOTE_WINDOW = 20        # גודל חלון להתכנסות
PROMOTE_MARGIN = 0.01      # ≥1% שיפור כדי לקדם
TAIL_STRICT = 5            # זנב לא עולה ב-5 צעדים אחרונים

def learn_from_pipeline_result(spec: Dict[str,Any], ab_decision: Dict[str,Any], *, user_id: str) -> Dict[str,Any]:
    """
    נקראת אחרי בחירת A/B. רושמת היסטוריה, בודקת התכנסות ורגרסיה,
    ומקדמת baseline אם שיפור "בטוח".
    """
    name = str(spec["name"]); goal = str(spec["goal"])
    key = _task_key(name, goal)

    phi = float(ab_decision["info"]["phi"])
    label = str(ab_decision["winner"]["label"])
    metrics = {
        "phi": phi,
        "p95_ms": float(ab_decision["info"]["perf"]["p95_ms"]),
        "error_rate": float(ab_decision["info"]["perf"]["error_rate"]),
        "cost_units": float(ab_decision["info"]["perf"]["cost_units"]),
        "label": label
    }

    # 1) היסטוריה
    append_history(key, dict(metrics, user_id=user_id))
    hist = load_history(key, limit=500)
    xs = [float(h["phi"]) for h in hist]

    # 2) baseline קיים?
    baseline = load_baseline(key)
    if baseline is None:
        # אימוץ ראשוני — התחלה מאפס
        save_baseline(key, dict(metrics, adopted_ts=time.time()))
        current().add_evidence("learn_update", {
            "source_url": f"local://learn/{key}",
            "trust": 0.95,
            "ttl_s": 7*24*3600,
            "payload": {"action":"adopt_initial", "metrics": metrics}
        })
        return {"adopted":"initial","baseline":metrics}

    # 3) בדיקת רגרסיה — נדרש שיפור לעומת baseline
    phi_base = float(baseline["phi"])
    can_promote = regression_guard(phi, phi_base, promote_margin=PROMOTE_MARGIN)

    # 4) בדיקת התכנסות אמפירית בחלון אחרון
    converged = has_converged(xs, window=PROMOTE_WINDOW, rel_tol=PROMOTE_MARGIN, strict_tail=TAIL_STRICT)

    if can_promote and converged:
        save_baseline(key, dict(metrics, adopted_ts=time.time(), prev=baseline))
        current().add_evidence("learn_update", {
            "source_url": f"local://learn/{key}",
            "trust": 0.95,
            "ttl_s": 7*24*3600,
            "payload": {"action":"promote", "new": metrics, "old": baseline, "window": PROMOTE_WINDOW}
        })
        return {"adopted":"promote","baseline":metrics}
    else:
        current().add_evidence("learn_update", {
            "source_url": f"local://learn/{key}",
            "trust": 0.8 if not can_promote else 0.9,
            "ttl_s": 3*24*3600,
            "payload": {"action":"hold", "reason":{
                "regression_ok": can_promote, "converged": converged,
                "phi_new": phi, "phi_baseline": phi_base
            }}
        })
        return {"adopted":"hold","baseline":baseline}