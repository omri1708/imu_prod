# imu_repo/engine/respond_bridge.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Callable
from engine.proof_chain import emit_with_proof
from engine.respond_guard import RespondBlocked
from engine.audit_log import record_event

def _extract_ctx_claims(ctx: Optional[Dict[str,Any]]) -> List[Dict[str,Any]]:
    """
    מאפשר תאימות לאחור: אם ה-VM/הפייפליין שמרו טענות ב-ctx["__claims__"],
    נחלץ אותן. אחרת נחזיר רשימה ריקה.
    Claim schema:
      {"id": str, "text": str, "evidence": [ { "kind":"inline", "content":... } | { "kind":"http","url":... } ]}
    """
    if not isinstance(ctx, dict): 
        return []
    claims = ctx.get("__claims__")
    if isinstance(claims, list):
        # מסננים אלמנטים לא-תקינים באופן שמרני (נחסום אח"כ אם לא תקין)
        out: List[Dict[str,Any]] = []
        for c in claims:
            if isinstance(c, dict) and "text" in c:
                # אם חסר id – נייצר דטרמיניסטית מפרגמנט הטקסט
                cid = c.get("id")
                if not isinstance(cid, str) or not cid:
                    import hashlib
                    cid = hashlib.sha256(str(c.get("text","")).encode("utf-8")).hexdigest()[:12]
                ev = c.get("evidence")
                out.append({"id": cid, "text": c.get("text",""), "evidence": ev if isinstance(ev, list) else []})
        return out
    return []

def _default_transport(text: str, meta: Dict[str,Any]) -> Dict[str,Any]:
    """
    טרנספורט ברירת-מחדל: *לא* שולח לרשת; רק מחזיר את המטא־דטה.
    אפשר להחליף בפועל ל-WebSocket/HTTP וכו' בהתאם לצורך.
    """
    return {"delivered": True, "meta": meta, "len": len(text)}

def respond_with_required_proof(
    *,
    response_text: str,
    ctx: Optional[Dict[str,Any]] = None,
    extra_claims: Optional[List[Dict[str,Any]]] = None,
    policy: Optional[Dict[str,Any]] = None,
    http_fetcher: Optional[Callable[[str,str], tuple]] = None,
    transport: Optional[Callable[[str, Dict[str,Any]], Any]] = None
) -> Dict[str,Any]:
    """
    נקודת ה-RESPOND המחייבת הוכחות:
      1) מאחדת claims מהקונטקסט ומ-extra_claims (אם ניתנו).
      2) קוראת ל-emit_with_proof (שבסופו של דבר סוגר CAS+Audit ומחסום הראיות).
      3) מחזירה {ok, proof_hash, response_hash, proof, transport_result?}
    אם המדיניות מחייבת ראיות – ייזרק RespondBlocked בהיעדר/פסילת ראיות.
    """
    p = dict(policy or {})
    # ברירת מחדל: מחייבים claims עבור כל תגובה (אפשר לשנות במדיניות)
    p.setdefault("require_claims_for_all_responses", True)
    # מותר להגדיר trusted_domains/max_http_age_days/http_download_for_hash במדיניות

    claims_ctx = _extract_ctx_claims(ctx)
    claims_ext = extra_claims if isinstance(extra_claims, list) else []
    claims: List[Dict[str,Any]] = [*claims_ctx, *claims_ext]

    try:
        pack = emit_with_proof(
            response_text=response_text,
            claims=claims,
            policy=p,
            transport=(transport or _default_transport),
            http_fetcher=http_fetcher
        )
        # רישום אירוע הצלחה מסומן (תיעוד שקוף)
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