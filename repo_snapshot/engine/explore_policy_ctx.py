# imu_repo/engine/explore_policy_ctx.py
from __future__ import annotations
import time
from typing import Dict, Any, List
from engine.explore_state import load_state, in_cooldown

DEFAULT_BY_INTENT = {
    "realtime":  {"base": 0.05, "min": 0.0,  "max": 0.2},
    "sensitive": {"base": 0.02, "min": 0.0,  "max": 0.1},
    "batch":     {"base": 0.2,  "min": 0.0,  "max": 0.6},
    "cost_saver":{"base": 0.4,  "min": 0.05, "max": 0.9},
    "gpu":       {"base": 0.15, "min": 0.0,  "max": 0.5},
    "mobile":    {"base": 0.1,  "min": 0.0,  "max": 0.4},
    "ui":        {"base": 0.15, "min": 0.0,  "max": 0.5},
}

def _blend(vals: List[float]) -> float:
    if not vals: 
        return 0.0
    return sum(vals) / float(len(vals))

def decide_explore_ctx(*, key: str, intents: List[str], history_len: int, cfg: Dict[str,Any]) -> bool:
    """
    מחזיר True אם כדאי לבצע Explore עבור המשימה.
    לוגיקה:
      1) אם ב-cooldown → False.
      2) קובע ε לפי Intent (ממוצע בין כמה תגים), מאפשר התאמות מה-config.
      3) ככל שהיסטוריה קטנה → מגדיל ε (חימום); ככל שגדולה → מצמצם מעט.
      4) מגביל לפי min/max intent.
    """
    # 1) Cooldown
    if in_cooldown(key):
        return False

    ex_cfg = dict(cfg.get("explore", {}))
    by_intent = dict(ex_cfg.get("by_intent", DEFAULT_BY_INTENT))
    epsilons = []
    mins, maxs = [], []
    for tag in intents or ["batch"]:
        row = by_intent.get(tag, DEFAULT_BY_INTENT.get(tag, {"base":0.1,"min":0.0,"max":0.5}))
        epsilons.append(float(row.get("base", 0.1)))
        mins.append(float(row.get("min", 0.0)))
        maxs.append(float(row.get("max", 0.5)))
    base_eps = _blend(epsilons)
    min_eps = max(0.0, _blend(mins))
    max_eps = max(base_eps, _blend(maxs))

    # 3) התאמת היסטוריה: מעט יותר אגרסיבי אם אין דגימות
    if history_len < 3:
        base_eps *= 1.8
    elif history_len < 10:
        base_eps *= 1.2
    elif history_len > 50:
        base_eps *= 0.8

    # גבולות סופיים
    eps = max(min_eps, min(max_eps, base_eps))

    # 4) הטלת קוביה – דטרמיניסטיות־לבדיקה: מועתק ל־hash הזמן (שומר על פשטות)
    import time
    t = int(time.time() * 997)  # מספר ראשוני
    # מוודאים התפלגות בינארית פשוטה:
    return (t % 1000) < int(eps * 1000.0 + 0.5)

# TODO- שים לב: המדיניות דטרמיניסטית מספיק לטסטים (תלויה בזמן). אם תרצה, אפשר להחליף למחולל פסאודו־אקראי עם seed.