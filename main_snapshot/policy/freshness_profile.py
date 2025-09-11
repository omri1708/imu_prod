# imu_repo/policy/freshness_profiles.py
from __future__ import annotations
from typing import Dict, Any

# פרופילים קשיחים המגדירים תקרת TTL ו"דעיכה יומית" בביטחון (אמון).
# ניתן להרחיב/לשנות בפריסה, אך כאן יש ברירות מחדל סבירות.
PROFILES: Dict[str, Dict[str, Any]] = {
    # דאטה דינמי מאוד
    "news":     {"max_ttl_s": 3*3600,     "decay_per_day": 0.30},
    "pricing":  {"max_ttl_s": 6*3600,     "decay_per_day": 0.25},
    "metrics":  {"max_ttl_s": 12*3600,    "decay_per_day": 0.20},
    # דאטה בינוני
    "docs":     {"max_ttl_s": 7*24*3600,  "decay_per_day": 0.08},
    "schema":   {"max_ttl_s": 14*24*3600, "decay_per_day": 0.05},
    "code":     {"max_ttl_s": 30*24*3600, "decay_per_day": 0.04},
    # זהויות/קונפיג רגישים — קצרי טווח בפרוד
    "identity": {"max_ttl_s": 24*3600,    "decay_per_day": 0.18},
    "config":   {"max_ttl_s": 72*3600,    "decay_per_day": 0.12},
    # חומרים יחסית יציבים
    "model":    {"max_ttl_s": 60*24*3600, "decay_per_day": 0.03},
    "ui":       {"max_ttl_s": 30*24*3600, "decay_per_day": 0.04},
    # ברירת־מחדל
    "default":  {"max_ttl_s": 7*24*3600,  "decay_per_day": 0.10},
}

def get_profile(kind: str | None) -> Dict[str,Any]:
    if not kind: return PROFILES["default"]
    return PROFILES.get(kind, PROFILES["default"])