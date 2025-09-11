# imu_repo/engine/phi_budget.py
from __future__ import annotations
from typing import Dict, Any
from engine.config import load_config, save_config
from engine.policy_ctx import get_user
from grounded.claims import current

def _phi_cfg() -> Dict[str, Any]:
    cfg = load_config()
    phi = dict(cfg.get("phi", {}))
    phi.setdefault("max_allowed", 50_000.0)
    phi.setdefault("per_capability_cost", {})  # לדוגמה: {"text.gen": 12.0}
    # נשמור חזרה (idempotent)
    cfg["phi"] = phi
    save_config(cfg)
    return phi

def available(user_id: str | None = None) -> float:
    phi = _phi_cfg()
    # בגרסה הזאת — תקציב משותף; אפשר להרחיב בעתיד לפר־משתמש
    return float(phi["max_allowed"])

def cost_for(capability: str, default: float = 1.0) -> float:
    phi = _phi_cfg()
    return float(phi["per_capability_cost"].get(capability, default))

def consume(capability: str, amount: float | None = None, *, user_id: str | None = None) -> tuple[float, float]:
    uid = user_id or (get_user() or "anon")
    phi = _phi_cfg()
    req = float(amount if amount is not None else cost_for(capability, default=1.0))
    have = float(phi["max_allowed"])
    if req > have:
        # Evidence: חריגת תקציב
        current().add_evidence("phi_reject", {
            "source_url": "local://phi",
            "trust": 0.95,
            "ttl_s": 600,
            "payload": {"user": uid, "cap": capability, "needed": req, "available": have}
        })
        from engine.errors import BudgetExceeded
        raise BudgetExceeded(capability, req, have)
    # מחייבים ומפחיתים
    phi["max_allowed"] = have - req
    save_config({"phi": phi})
    current().add_evidence("phi_charge", {
        "source_url": "local://phi",
        "trust": 0.98,
        "ttl_s": 600,
        "payload": {"user": uid, "cap": capability, "charged": req, "remaining": float(phi["max_allowed"])}
    })
    return req, float(phi["max_allowed"])