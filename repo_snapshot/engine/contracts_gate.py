# engine/contracts_gate.py
# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Optional
import time
from governance.policy import RespondPolicy
from grounded.evidence_contracts import EvidenceIndex, Evidence, compute_sha256
from contracts.errors import ContractViolation
from common.errors import ContractError, EvidenceMissing
from evidence import cas


class ContractViolation(Exception): ...
class PolicyDenied(Exception): ...

def _type_ok(v, t: str) -> bool:
    if t == "int": return isinstance(v, int) and not isinstance(v, bool)
    if t == "float": return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == "str": return isinstance(v, str)
    if t == "bool": return isinstance(v, bool)
    if t == "list": return isinstance(v, list)
    if t == "dict": return isinstance(v, dict)
    return False


def _check_schema(val, schema: Dict[str, Any], path: str):
    t = schema.get("type")
    if t and not _type_ok(val, t):
        raise ContractError(f"schema_type_mismatch at {path}: want {t}, got {type(val).__name__}")
    if t in ("int","float"):
        lo = schema.get("min"); hi = schema.get("max")
        if lo is not None and val < lo: raise ContractError(f"min_violation at {path}: {val}<{lo}")
        if hi is not None and val > hi: raise ContractError(f"max_violation at {path}: {val}>{hi}")
        unit = schema.get("unit")
        if unit and not isinstance(unit, str):
            raise ContractError(f"unit_malformed at {path}")
    if t == "list":
        es = schema.get("elements")
        if es:
            for i, vv in enumerate(val):
                _check_schema(vv, es, f"{path}[{i}]")
    if t == "dict":
        props = schema.get("properties", {})
        for k, ss in props.items():
            if k not in val: raise ContractError(f"missing_property {path}.{k}")
            _check_schema(val[k], ss, f"{path}.{k}")


def _evidence_ok(e: Dict[str, Any], *, now: float, trust_threshold: float) -> bool:
    sha = e.get("sha256"); ttl = e.get("ttl_sec", 365*24*3600)
    fetched = e.get("fetched_at", now)
    if not isinstance(sha, str): return False
    if cas.get(sha) is None: return False
    age = max(0.0, now - float(fetched))
    if age > float(ttl): return False
    trust = float(e.get("trust", 0.5))
    if trust < trust_threshold: return False
    return True


def enforce_respond_contract(*, stage: str, claims: List[Dict[str, Any]], evidence: List[Dict[str, Any]],
                             policy, ev_index):
    """
    חוזה קשיח: לכל claim חייבת להיות לפחות ראיה אחת תקפה ב-CAS,
    והערך עומד בסכימה/טווחים/יחידות. אחרת: ContractError/EvidenceMissing.
    """
    now = time.time()
    # נבנה אינדקס ראיות לפי sha256
    ev_map = {}
    for e in evidence or []:
        sha = e.get("sha256")
        if not sha: continue
        sc = ev_index.score(e)  # משלב trust חיצוני אם קיים
        e = dict(e); e["trust"] = max(e.get("trust", 0.0), sc)
        ev_map.setdefault(sha, []).append(e)

    for c in claims or []:
        path = c.get("id","claim")
        val  = c.get("value", None)
        schema = c.get("schema", {})
        _check_schema(val, schema, path)

        ev_list = c.get("evidence", [])
        ok = False
        for e in ev_list:
            sha = e.get("sha256")
            if not sha: continue
            for ee in ev_map.get(sha, []):
                if _evidence_ok(ee, now=now, trust_threshold=policy.trust_threshold):
                    ok = True; break
            if ok: break

        if not ok:
            # אם יש ראיות אך לא תקינות – EvidenceMissing; אם אין בכלל – ContractError
            if ev_list:
                raise EvidenceMissing(f"invalid_stale_or_untrusted_evidence for {path}")
            raise ContractError(f"missing_evidence for {path}")       


def attach_claim(text: str, source_bytes: bytes, url: str, trust: float, sig_ok: bool, ev_index: EvidenceIndex) -> Dict[str,Any]:
    sha = compute_sha256(source_bytes)
    rec = {"sha256":sha,"ts":0,"trust":trust,"url":url,"sig_ok":sig_ok}
    ev_index.put(sha, {"ts":0,"trust":trust,"url":url,"sig_ok":sig_ok})
    return rec


def _is_math_expression(text: str) -> bool:
    # זיהוי מאוד שמרני (ללא צד שלישי) – ביטוי מספרי פשוט
    import re
    return bool(re.fullmatch(r"[0-9\.\+\-\*\/\(\) \t]+", text or ""))


 