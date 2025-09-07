from __future__ import annotations
from typing import Any, Dict
from engine.audit_log import record_event

class EmitBlocked(Exception): ...

def _get(d: dict, path: str, default=None):
    cur = d
    for p in path.split("."):
        if not isinstance(cur, dict): return default
        cur = cur.get(p, default if p == path.split(".")[-1] else {})
    return cur

def _enforce(ctx: Dict[str,Any]) -> None:
    # 1) Gate after חייב להיות קיים ו-ok
    ga = ctx.get("gate",{}).get("after")
    if not ga:
        raise EmitBlocked("missing gate.after")
    # 2) min_trust אפקטיבי
    min_trust = float(_get(ga, "policy.min_trust", 0.75))
    # 3) Provenance של החבילה קיים
    prov = (ctx.get("package") or {}).get("provenance") or {}
    agg  = prov.get("agg_trust", min_trust)
    if agg < min_trust:
        raise EmitBlocked(f"agg_trust {agg:.2f} < min_trust {min_trust:.2f}")
    # 4) אפשרויות נוספות: לאשר רק outputs חתומים, לבדוק KPI ok:
    guard = ctx.get("guard") or {}
    kpi_ok = bool((guard.get("kpi") or {}).get("ok", True))
    if not kpi_ok:
        raise EmitBlocked("kpi_guard_failed")

def agent_emit_answer(*, answer_text: str, ctx: Dict[str,Any], policy: Dict[str,Any]) -> Dict[str,Any]:
    _enforce(ctx)
    # לוג Audit
    rec = record_event("agent_emit_answer", {
        "len": len(answer_text),
        "user_id": ctx.get("user_id"),
        "domain": ctx.get("domain"),
        "risk": ctx.get("risk_hint"),
        "artifact_sha": (ctx.get("package") or {}).get("provenance", {}).get("artifact_sha"),
        "manifest_sha": (ctx.get("package") or {}).get("provenance", {}).get("manifest_sha"),
    }, severity="info")
    return {
        "ok": True,
        "answer": answer_text,
        "audit_id": rec["id"],
        "artifact": (ctx.get("package") or {}).get("provenance", {})
    }
