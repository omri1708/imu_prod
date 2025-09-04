# imu_repo/engine/respond_guard.py
from __future__ import annotations
import time
import email.utils as eutils
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from engine.cas_store import put_json, put_bytes
from engine.audit_log import record_event

class RespondBlocked(Exception): ...

def _now_ts() -> float:
    return time.time()

def _parse_date_header(h: Optional[str]) -> Optional[float]:
    if not h:
        return None
    try:
        dt = eutils.parsedate_to_datetime(h)
        return dt.timestamp() if dt else None
    except Exception:
        return None

def _host_allowed(url: str, trusted: Optional[List[str]]) -> bool:
    if not trusted:
        return True
    host = urlparse(url).hostname or ""
    return any(host == t or host.endswith("." + t) for t in trusted)

def _validate_claims_structure(claims: List[Dict[str,Any]]) -> None:
    if not isinstance(claims, list) or len(claims) == 0:
        raise RespondBlocked("claims_required: no claims provided")
    for i, c in enumerate(claims):
        if not isinstance(c, dict):
            raise RespondBlocked(f"claim#{i} invalid: not dict")
        if not c.get("id") or not isinstance(c.get("id"), str):
            raise RespondBlocked(f"claim#{i} invalid: missing id")
        if not isinstance(c.get("text"), str) or not c["text"]:
            raise RespondBlocked(f"claim#{i} invalid: missing text")
        ev = c.get("evidence")
        if not isinstance(ev, list) or len(ev) == 0:
            raise RespondBlocked(f"claim#{i} invalid: missing evidence")

def _evidence_inline_pack(e: Dict[str,Any]) -> Dict[str,Any]:
    content = e.get("content")
    if not isinstance(content, (str, bytes)) or (isinstance(content, str) and content == ""):
        raise RespondBlocked("evidence_inline_invalid: missing content")
    if isinstance(content, str):
        content_b = content.encode("utf-8")
    else:
        content_b = content
    h = put_bytes(content_b)
    return {"kind":"inline","hash":h,"bytes":len(content_b)}

def _evidence_http_pack(
    e: Dict[str,Any],
    *,
    policy: Dict[str,Any],
    http_fetcher=None   # Optional[(url, method) -> (status:int, headers:dict, body:bytes|None)]
) -> Dict[str,Any]:
    url = e.get("url")
    if not isinstance(url, str) or not url.startswith(("http://","https://")):
        raise RespondBlocked("evidence_http_invalid: bad url")
    trusted = policy.get("trusted_domains")  # list[str] or None
    if not _host_allowed(url, trusted):
        raise RespondBlocked(f"evidence_http_untrusted_host: {url}")
    # אימות בסיסי: HEAD (או GET לפי מדיניות)
    method = "HEAD"
    must_download = bool(policy.get("http_download_for_hash", False))
    if must_download:
        method = "GET"
    # fetch (מזריק בבדיקות; בפרודקשן אפשר להשתמש urllib.request)
    status, headers, body = (None, None, None)
    if http_fetcher is not None:
        status, headers, body = http_fetcher(url, method)
    else:
        # מימוש stdlib: urllib
        import urllib.request
        req = urllib.request.Request(url, method=method)
        with urllib.request.urlopen(req, timeout=float(policy.get("http_timeout_sec", 5.0))) as resp:
            status = resp.status
            headers = {k.lower(): v for k,v in resp.getheaders()}
            body = resp.read() if must_download else None
    if not (200 <= int(status) < 400):
        raise RespondBlocked(f"evidence_http_status_not_ok: {status} for {url}")
    # בדיקת עדכניות (אופציונלי)
    max_age_days = policy.get("max_http_age_days")
    if isinstance(max_age_days, (int,float)):
        dt = _parse_date_header((headers or {}).get("date"))
        if dt is not None and (_now_ts() - dt) > (float(max_age_days)*86400.0):
            raise RespondBlocked(f"evidence_http_stale: {url}")
    meta = {"url": url, "status": int(status), "headers": headers or {}}
    meta_hash = put_json(meta)
    body_hash = None
    if must_download and body is not None:
        body_hash = put_bytes(body)
    return {"kind":"http","meta_hash":meta_hash, "body_hash":body_hash}

def _pack_evidence_list(
    claims: List[Dict[str,Any]],
    *,
    policy: Dict[str,Any],
    http_fetcher=None
) -> Tuple[List[Dict[str,Any]], Dict[str,Any]]:
    """
    מחזיר (packed_evidence, map claim_id -> indices של evidence ארוז)
    """
    packed: List[Dict[str,Any]] = []
    cmap: Dict[str,Any] = {}
    for c in claims:
        cid = c["id"]
        cmap[cid] = {"evidence_idx": []}
        for ev in c["evidence"]:
            kind = ev.get("kind")
            if kind == "inline":
                pe = _evidence_inline_pack(ev)
            elif kind == "http":
                pe = _evidence_http_pack(ev, policy=policy, http_fetcher=http_fetcher)
            else:
                raise RespondBlocked(f"evidence_unknown_kind: {kind}")
            cmap[cid]["evidence_idx"].append(len(packed))
            packed.append(pe)
    return packed, cmap

def ensure_proof_and_package(
    *,
    response_text: str,
    claims: List[Dict[str,Any]],
    policy: Dict[str,Any],
    http_fetcher=None
) -> Dict[str,Any]:
    """
    אוכף:
    - חובה claims (אם policy['require_claims_for_all_responses']=True)
    - כל evidence תקין; HTTP מאומת; CAS hashes נשמרים
    - חותך bundle הוכחות (proof) ושומר ל-CAS
    מחזיר: {ok, proof_hash, proof, response_hash}
    """
    if bool(policy.get("require_claims_for_all_responses", True)):
        _validate_claims_structure(claims)
    else:
        # אם לא מחייבים claims — נאפשר תשובה גם בלעדיהם
        if not claims:
            # בכל זאת נארוז חבילת הוכחות ריקה
            bundle = {"version":1,"claims":[], "evidence":[], "map":{}, "ts": _now_ts()}
            proof_hash = put_json(bundle)
            resp_hash = put_bytes(response_text.encode("utf-8"))
            record_event("respond_proof_ok", {"claims":0,"evidence":0,"proof_hash":proof_hash,"response_hash":resp_hash}, severity="info")
            return {"ok": True, "proof_hash": proof_hash, "proof": bundle, "response_hash": resp_hash}

    packed, cmap = _pack_evidence_list(claims, policy=policy, http_fetcher=http_fetcher)
    bundle = {
        "version": 1,
        "ts": _now_ts(),
        "claims": [{"id":c["id"], "text":c["text"]} for c in claims],
        "evidence": packed,
        "map": cmap
    }
    proof_hash = put_json(bundle)
    resp_hash = put_bytes(response_text.encode("utf-8"))
    record_event("respond_proof_ok", {
        "claims": len(bundle["claims"]),
        "evidence": len(bundle["evidence"]),
        "proof_hash": proof_hash,
        "response_hash": resp_hash
    }, severity="info")
    return {"ok": True, "proof_hash": proof_hash, "proof": bundle, "response_hash": resp_hash}