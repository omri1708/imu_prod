# imu_repo/engine/respond_guard.py  (גרסה מעודכנת עם Provenance)
from __future__ import annotations
import time
import email.utils as eutils
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from engine.cas_store import put_json, put_bytes
from engine.audit_log import record_event
from engine.trust_tiers import enforce_trust_requirements, TrustPolicyError
from engine.consistency import validate_claims_and_consistency, ConsistencyError
from engine.provenance import enforce_min_provenance, ProvenanceError
from engine.cas_sign import sign_manifest
from engine.consistency_graph_weighted import WeightedConsistencyGraph, WeightedConsistencyError

class RespondBlocked(Exception): ...


def _weighted_consistency_if_requested(claims: List[Dict[str,Any]], policy: Dict[str,Any]) -> None:
    gspec = policy.get("global_consistency")  # {"relations":[{"a":"modA:x","b":"modB:x","rel":"equal","tol_pct":0.05,"dominates":"a"}], "weights":{"modA:x":3.0,...}}
    if not isinstance(gspec, dict):
        return
    rels = gspec.get("relations") or []
    wmap = gspec.get("weights") or {}
    if not isinstance(rels, list):
        return
    G = WeightedConsistencyGraph()
    # נבנה אינדקס טענות לפי id מלא
    idx = {c["id"]: c for c in claims if isinstance(c.get("id"), str)}
    for cid, c in idx.items():
        w = float(wmap.get(cid, 1.0))
        G.add_claim(cid, c, weight=w)
    for r in rels:
        a = r.get("a"); b = r.get("b"); rel = r.get("rel")
        meta = dict(r); meta.pop("a",None); meta.pop("b",None); meta.pop("rel",None)
        if isinstance(a,str) and isinstance(b,str) and isinstance(rel,str):
            G.relate(a,b,rel, **meta)
    G.check()

    
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
    http_fetcher=None
) -> Dict[str,Any]:
    url = e.get("url")
    if not isinstance(url, str) or not url.startswith(("http://","https://")):
        raise RespondBlocked("evidence_http_invalid: bad url")
    trusted = policy.get("trusted_domains")  # list[str] or None
    if not _host_allowed(url, trusted):
        raise RespondBlocked(f"evidence_http_untrusted_host: {url}")

    method = "HEAD"
    must_download = bool(policy.get("http_download_for_hash", False))
    if must_download:
        method = "GET"

    status, headers, body = (None, None, None)
    if http_fetcher is not None:
        status, headers, body = http_fetcher(url, method)
    else:
        import urllib.request
        req = urllib.request.Request(url, method=method)
        with urllib.request.urlopen(req, timeout=float(policy.get("http_timeout_sec", 5.0))) as resp:
            status = resp.status
            headers = {k.lower(): v for k,v in resp.getheaders()}
            body = resp.read() if must_download else None

    if not (200 <= int(status) < 400):
        raise RespondBlocked(f"evidence_http_status_not_ok: {status} for {url}")

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
    return {"kind":"http","meta_hash":meta_hash, "body_hash":body_hash, "url": url}

def _pack_evidence_list(
    claims: List[Dict[str,Any]],
    *,
    policy: Dict[str,Any],
    http_fetcher=None
) -> Tuple[List[Dict[str,Any]], Dict[str,Any]]:
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

def _apply_trust_and_consistency(claims: List[Dict[str,Any]], policy: Dict[str,Any]) -> None:
    # אכיפת Trust למקור עבור כל claim
    for c in claims:
        enforce_trust_requirements(c, policy)

    # אכיפת סכימות ו-Consistency בין טענות
    validate_claims_and_consistency(
        claims,
        require_consistency_groups=bool(policy.get("require_consistency_groups", False)),
        default_number_tolerance=float(policy.get("default_number_tolerance", 0.01))
    )

def ensure_proof_and_package(
    *,
    response_text: str,
    claims: List[Dict[str,Any]],
    policy: Dict[str,Any],
    http_fetcher=None
) -> Dict[str,Any]:
    need_claims = bool(policy.get("require_claims_for_all_responses", True))
    if need_claims:
        _validate_claims_structure(claims)
    elif not claims:
        bundle = {"version":1,"claims":[], "evidence":[], "map":{}, "ts": time.time()}
        proof_hash = put_json(bundle)
        resp_hash = put_bytes(response_text.encode("utf-8"))
        record_event("respond_proof_ok", {"claims":0,"evidence":0,"proof_hash":proof_hash,"response_hash":resp_hash}, severity="info")
        return {"ok": True, "proof_hash": proof_hash, "proof": bundle, "response_hash": resp_hash}

    try:
        # Trust + Consistency (מקומי)
        validate_claims_and_consistency(
            claims,
            require_consistency_groups=bool(policy.get("require_consistency_groups", False)),
            default_number_tolerance=float(policy.get("default_number_tolerance", 0.01))
        )
        for c in claims:
            enforce_trust_requirements(c, policy)
            enforce_min_provenance(c, policy)
        # Consistency גלובלי משוקלל (אם נדרש במדיניות)
        _weighted_consistency_if_requested(claims, policy)
    except (TrustPolicyError, ConsistencyError, ProvenanceError, WeightedConsistencyError) as e:
        raise RespondBlocked(str(e))

    packed, cmap = _pack_evidence_list(claims, policy=policy, http_fetcher=http_fetcher)
    bundle = {
        "version": 3,
        "ts": time.time(),
        "claims": [{"id":c["id"], "text":c["text"], "schema": c.get("schema"), "value": c.get("value"), "group": c.get("consistency_group"), "type": c.get("type")} for c in claims],
        "evidence": packed,
        "map": cmap
    }
    # חתימת CAS אם מוגדר מפתח
    sk = policy.get("signing_keys") or {}
    default_kid = next(iter(sk.keys()), None)
    if default_kid:
        meta = sk[default_kid]
        sig = sign_manifest(bundle, key_id=default_kid, secret_hex=str(meta["secret_hex"]), algo=str(meta.get("algo","sha256")))
        bundle["signature"] = sig

    proof_hash = put_json(bundle)
    resp_hash = put_bytes(response_text.encode("utf-8"))
    record_event("respond_proof_ok", {
        "claims": len(bundle["claims"]),
        "evidence": len(bundle["evidence"]),
        "proof_hash": proof_hash,
        "response_hash": resp_hash
    }, severity="info")
    return {"ok": True, "proof_hash": proof_hash, "proof": bundle, "response_hash": resp_hash}