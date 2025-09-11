# imu_repo/engine/respond_bridge.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Callable
from engine.respond_guard import ensure_proof_and_package, RespondBlocked
from engine.audit_log import record_event

def _extract_ctx_claims(ctx: Optional[Dict[str,Any]]) -> List[Dict[str,Any]]:
    if not isinstance(ctx, dict):
        return []
    claims = ctx.get("__claims__")
    if isinstance(claims, list):
        out: List[Dict[str,Any]] = []
        for c in claims:
            if isinstance(c, dict) and "text" in c:
                cid = c.get("id")
                if not isinstance(cid, str) or not cid:
                    import hashlib
                    cid = hashlib.sha256(str(c.get("text","")).encode("utf-8")).hexdigest()[:12]
                ev = c.get("evidence")
                out.append({"id": cid, "text": c.get("text",""), "evidence": ev if isinstance(ev, list) else []})
        return out
    return []

def _default_transport(text: str, meta: Dict[str,Any]) -> Dict[str,Any]:
    return {"delivered": True, "meta": meta, "len": len(text)}

def _resolve_http_fetcher(
    *,
    explicit_fetcher: Optional[Callable[[str,str], tuple]],
    ctx: Optional[Dict[str,Any]],
    policy: Dict[str,Any],
) -> Optional[Callable[[str,str], tuple]]:
    """
    קדימות:
      1) explicit_fetcher שנמסר ישירות לפונקציה
      2) ctx["__http_fetcher__"] אם הוזרק בסביבה
      3) policy["http_fetcher"] אם יש (למשל wiring חיצוני)
      4) None (ללא רשת; בדיקות/CI)
    """
    if explicit_fetcher is not None:
        return explicit_fetcher
    if isinstance(ctx, dict):
        f = ctx.get("__http_fetcher__")
        if callable(f):
            return f
    f2 = policy.get("http_fetcher")
    if callable(f2):
        return f2
    return None

def respond_with_required_proof(
    *,
    response_text: str,
    ctx: Optional[Dict[str,Any]] = None,
    extra_claims: Optional[List[Dict[str,Any]]] = None,
    policy: Optional[Dict[str,Any]] = None,
    http_fetcher: Optional[Callable[[str,str], tuple]] = None,
    transport: Optional[Callable[[str, Dict[str,Any]], Any]] = None
) -> Dict[str,Any]:
    p = dict(policy or {})
    p.setdefault("require_claims_for_all_responses", True)

    claims_ctx = _extract_ctx_claims(ctx)
    claims_ext = extra_claims if isinstance(extra_claims, list) else []
    claims: List[Dict[str,Any]] = [*claims_ctx, *claims_ext]

    fetcher = _resolve_http_fetcher(explicit_fetcher=http_fetcher, ctx=ctx, policy=p)

    try:
        pack = ensure_proof_and_package(
            response_text=response_text,
            claims=claims,
            policy=p,
            transport=(transport or _default_transport),
            http_fetcher=fetcher
        )
        record_event("respond_finalized", {
            "claims_count": len(pack["proof"]["claims"]),
            "evidence_count": len(pack["proof"]["evidence"]),
            "proof_hash": pack["proof_hash"],
            "response_hash": pack["response_hash"]
        }, severity="info")
        return {"ok": True, **pack}
    except RespondBlocked as e:
        record_event("respond_blocked", {"reason": str(e)}, severity="warning")
        return {"ok": False, "error": str(e)}