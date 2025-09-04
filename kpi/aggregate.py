# imu_repo/kpi/aggregate.py
from __future__ import annotations
from typing import Dict, Any, List

def _service_weight(summary: Dict[str,Any]) -> float:
    # משקל פר־שירות: קירוב לפי עומס/חשיבות — כאן: 60% ל-api, 40% ל-worker; אם אין – כולם שווים.
    name = (summary.get("generated",{}).get("service_name") or summary.get("generated",{}).get("language") or "")
    n = summary.get("generated",{}).get("name") or ""
    # אם בשם יש :api/:worker נשתמש בו
    if ":api" in (summary.get("generated",{}).get("name") or ""): 
        return 0.6
    if ":worker" in (summary.get("generated",{}).get("name") or ""):
        return 0.4
    return 1.0

def aggregate(components: List[Dict[str,Any]]) -> Dict[str,Any]:
    """
    מחזיר KPI משוקלל ורול־אאוט כולל:
      - חייב שכל תתי־השירותים approved
      - ציון כולל = ממוצע משוקלל
    """
    if not components:
        return {"approved": False, "reason":"no_components"}

    total_w, score = 0.0, 0.0
    all_approved = True
    parts=[]
    for c in components:
        k = (c.get("rollout") or {}).get("kpi") or (c.get("kpi") or {})
        s = float(k.get("score", 0.0))
        w = _service_weight(c)
        total_w += w
        score += s * w
        appr = bool((c.get("rollout") or {}).get("approved", False))
        all_approved = all_approved and appr
        parts.append({"name": c.get("generated",{}).get("name"), "score": s, "approved": appr})

    score = score / max(1e-9, total_w)
    return {"approved": all_approved, "score": score, "parts": parts}