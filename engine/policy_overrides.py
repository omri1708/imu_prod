# imu_repo/engine/policy_overrides.py
from __future__ import annotations
from typing import Dict, Any

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def deep_merge(dst: Dict[str,Any], src: Dict[str,Any]) -> Dict[str,Any]:
    out = dict(dst or {})
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def overrides_for_user(user: Dict[str,Any]) -> Dict[str,Any]:
    """
    גוזר overrides למדיניות לפי פרופיל המשתמש/קונטקסט.
    דוגמאות:
      - לקוח אנטרפרייז/רגיש → הקשחת evidences, p95 הדוק יותר, drift קטן יותר.
      - משתמש ניסויי → p95 מרווח יותר, מאפשר autotune מהיר יותר.
    """
    tier = (user or {}).get("tier") or "standard"
    risk = float((user or {}).get("risk_score", 0.5))
    # בסיס: עדכון ספי ביצועים ואמינות
    if tier in ("enterprise","regulated"):
        lat_p95 = _clamp(100.0 - 30.0*risk, 50.0, 100.0)   # מ״ש
        thr_min = _clamp(200.0 + 100.0*(1.0-risk), 200.0, 300.0)
        drift = _clamp(0.05 - 0.03*(1.0-risk), 0.01, 0.05)
        return {
            "min_distinct_sources": 2,
            "min_total_trust": 2,
            "perf_sla": {
                "latency_ms": {"p95_max": lat_p95},
                "throughput_rps": {"min": thr_min},
                "error_rate": {"max": 0.02},
                "near_miss_factor": 1.10
            },
            "consistency": {
                "drift_pct": drift,
                "near_miss_streak_heal_threshold": 2,
                "heal_action": "raise_require_fresh"
            },
        }
    elif tier in ("experimental","dev"):
        return {
            "min_distinct_sources": 1,
            "min_total_trust": 1,
            "perf_sla": {
                "latency_ms": {"p95_max": 250.0},
                "throughput_rps": {"min": 30.0},
                "error_rate": {"max": 0.10},
                "near_miss_factor": 1.35
            },
            "consistency": {
                "drift_pct": 0.25,
                "near_miss_streak_heal_threshold": 4,
                "heal_action": "freeze_autotune"
            },
        }
    else:  # standard
        return {
            "perf_sla": {
                "latency_ms": {"p95_max": 150.0},
                "throughput_rps": {"min": 100.0},
                "error_rate": {"max": 0.05},
                "near_miss_factor": 1.15
            },
            "consistency": {
                "drift_pct": 0.10,
                "near_miss_streak_heal_threshold": 3,
                "heal_action": "freeze_autotune"
            },
        }

def apply_user_overrides(base_policy: Dict[str,Any], user: Dict[str,Any]) -> Dict[str,Any]:
    """
    מחזיר policy ממוזג עם התאמות פר־משתמש/קונטקסט.
    """
    return deep_merge(base_policy or {}, overrides_for_user(user or {}))

def apply_overrides(policy: Dict[str, Any], *, channel: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    מחזיר policy עם התאמות לערוץ.
    ערוצים: "batch", "interactive", "realtime"
    """
    p = dict(policy or {})
    ch = channel.lower()
    # ברירת מחדל — חוקים משותפים
    p.setdefault("min_distinct_sources", 1)
    p.setdefault("min_total_trust", 1.0)
    sla = p.setdefault("perf_sla", {"latency_ms": {"p95_max": 200.0}})
    if ch == "realtime":
        # מחמירים ב-latency, שומרים על trust מינימלי
        sla["latency_ms"] = {"p95_max": 120.0}
        p["min_total_trust"] = max(p.get("min_total_trust", 1.0), 1.0)
    elif ch == "interactive":
        sla["latency_ms"] = {"p95_max": 250.0}
    elif ch == "batch":
        sla["latency_ms"] = {"p95_max": 5_000.0}

    # התאמות פר־משתמש (אם קיימות)
    user = (ctx or {}).get("user") or {}
    if user.get("tier") == "strict":
        p["min_distinct_sources"] = max(2, int(p.get("min_distinct_sources", 1)))
        p["min_total_trust"] = max(2.0, float(p.get("min_total_trust", 1.0)))
    return p