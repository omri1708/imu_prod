# imu_repo/user_model/policy.py
from __future__ import annotations
from typing import Dict, Any
from user_model.identity import load_policy, save_policy
from user_model.consciousness import route_decision

def effective_kpi(uid: str, *, default_p95_ms: float) -> Dict[str,Any]:
    pol = load_policy(uid)
    # adjust by "quality" + מודעות דינמית
    if pol.get("quality") == "strict":
        base = min(default_p95_ms, pol.get("latency_p95_ms", default_p95_ms))
    elif pol.get("quality") == "relaxed":
        base = max(default_p95_ms, pol.get("latency_p95_ms", default_p95_ms))
    else:
        base = pol.get("latency_p95_ms", default_p95_ms)
    routed = route_decision(uid, base_p95_ms=base)
    return {"p95_ms": routed["p95_target_ms"]}

def update_policy(uid: str, updates: Dict[str,Any]) -> Dict[str,Any]:
    pol = load_policy(uid)
    pol.update(updates or {})
    save_policy(uid, pol)
    return pol

# אחסון זכרון־תהליך (לצרכי טסטים). אפשר להחליף לשמירה לקובץ/DB.
_PROFILES: Dict[str, Dict[str,Any]] = {}

_DEFAULT = {
    "min_trust": 0.7,
    "max_age_s": 3600,
    "strict_grounded": True,
    "phi_weights": {
        "latency": 0.6,
        "cost": 0.25,
        "errors": 0.10,
        "distrust": 0.03,
        "energy": 0.015,
        "memory": 0.005,
    }
}

def set_profile(user_id: str, **kwargs) -> None:
    prof = dict(_PROFILES.get(user_id, _DEFAULT))
    # אפשר להציב משקולות דרך phi_weights=...
    if "phi_weights" in kwargs:
        ws = dict(_DEFAULT["phi_weights"])
        ws.update({k: float(v) for k,v in dict(kwargs["phi_weights"]).items()})
        prof["phi_weights"] = ws
        kwargs = {k:v for k,v in kwargs.items() if k != "phi_weights"}
    for k,v in kwargs.items():
        prof[k] = v
    _PROFILES[user_id] = prof

def get_profile(user_id: str) -> Dict[str,Any]:
    return dict(_PROFILES.get(user_id, _DEFAULT))