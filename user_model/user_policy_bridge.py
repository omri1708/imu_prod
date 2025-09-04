from __future__ import annotations
from typing import Dict, Any
from user_model.policy import effective_kpi

def apply_user_policy(uid: str, kpi_targets: Dict[str,Any]) -> Dict[str,Any]:
    """
    מקבל ספי KPI בסיסיים (למשל {"p95_ms":1500}) ומחזיר ספים מותאמים למשתמש.
    """
    base = dict(kpi_targets or {})
    p95 = float(base.get("p95_ms", 1500.0))
    eff = effective_kpi(uid, default_p95_ms=p95)
    base["p95_ms"] = eff["p95_ms"]
    return base