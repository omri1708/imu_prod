# imu_repo/safe_progress/auto_rollout.py
from __future__ import annotations
from typing import Dict, Any
from metrics.aggregate import aggregate_metrics
from sla.policy import SlaSpec, evaluate, compare

DEC_PROMOTE = "promote"
DEC_HOLD    = "hold"
DEC_ROLLBACK= "rollback"

def decide(*, window_s: int=600,
           name: str="guarded_handler",
           sla: SlaSpec | None=None,
           require_improvement: bool=False,
           min_rel_impr: float=0.05) -> Dict[str,Any]:
    """
    מחליט rollout אוטומטי עבור canary לעומת baseline:
      1) canary עומד ב-SLA קשיח (אם סופק).
      2) canary לא נחות מבייסליין (או משתפר אם require_improvement=True).
    """
    base = aggregate_metrics(name=name, bucket="baseline", window_s=window_s)
    can  = aggregate_metrics(name=name, bucket="canary",   window_s=window_s)

    sla_res = {"ok": True}
    if sla is not None:
        sla_res = evaluate(can, sla)

    cmp_res = compare(base, can, require_improvement=require_improvement, min_rel_impr=min_rel_impr)

    if not sla_res["ok"]:
        decision = DEC_ROLLBACK
    else:
        decision = DEC_PROMOTE if cmp_res["ok"] else DEC_HOLD

    return {
        "decision": decision,
        "sla": sla_res,
        "comparison": cmp_res,
        "baseline": base,
        "canary": can
    }