# imu_repo/user_model/policies.py
from __future__ import annotations
from typing import Dict, Any, Optional
import os, json

POLICY_PATH = "/mnt/data/imu_repo/policies/user_policies.json"

_DEFAULT = {
    "kpi_weights": { "tests":0.28, "latency":0.20, "ui":0.12, "consistency":0.28, "resolution":0.12 },
    "min_trust_by_key": {},                         # override למפתחות ראיה ספציפיים
    "min_consistency_score": 80.0,                  # סף עקביות מינימלי
    "trust_cut_for_resolution": 0.80,               # trust_cut ברזולוציית סתירות
    "canary_stages": None,                          # אם None → ברירת המחדל של canary_multi
    "anti_regression": { "max_regression_pct":7.5, "min_kpi":70.0 },
}

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _load() -> Dict[str, Any]:
    if not os.path.exists(POLICY_PATH):
        _ensure_dir(os.path.dirname(POLICY_PATH))
        with open(POLICY_PATH, "w", encoding="utf-8") as f:
            json.dump({"by_user":{}, "by_app":{}}, f, ensure_ascii=False, indent=2)
    with open(POLICY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _save(st: Dict[str,Any]):
    _ensure_dir(os.path.dirname(POLICY_PATH))
    with open(POLICY_PATH, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)

def set_for_user(user_id: str, policy: Dict[str,Any]) -> None:
    st = _load()
    st["by_user"][user_id] = policy
    _save(st)

def set_for_app(app: str, policy: Dict[str,Any]) -> None:
    st = _load()
    st["by_app"][app] = policy
    _save(st)

def get_effective(user_id: str, app: str) -> Dict[str,Any]:
    """
    קדימות: per-app → per-user → default.
    חיבור מפות (e.g. min_trust_by_key) נעשה במיזוג.
    """
    st = _load()
    eff = json.loads(json.dumps(_DEFAULT))  # deep copy
    by_app = st.get("by_app", {}).get(app) or {}
    by_user = st.get("by_user", {}).get(user_id) or {}

    # מיזוג פשוט: עליון גובר; מיפויים מתאחדים
    for src in (by_user, by_app):  # שים לב: app גובר מעל user? נהפוך — נסדר עדיפות app בסוף:
        pass
    # נעשה במפורש: קודם user, אחר-כך app (app עדיף)
    def _merge(dst: Dict[str,Any], src: Dict[str,Any]):
        for k,v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                dst[k].update(v)
            else:
                dst[k] = v
    _merge(eff, by_user)
    _merge(eff, by_app)
    return eff