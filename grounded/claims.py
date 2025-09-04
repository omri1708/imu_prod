# imu_repo/grounded/claims.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import threading, json

from grounded.provenance_store import add_evidence
from grounded.gate import require, GateDenied

_ctx_local = threading.local()


class _Claims:
    def __init__(self) -> None:
        self._buf: List[Tuple[str, Dict[str,Any]]] = []

    def add_evidence(self, key: str, ev: Dict[str,Any]) -> None:
        if not isinstance(key, str): 
            raise TypeError("key must be str")
        if not isinstance(ev, dict):
            raise TypeError("evidence must be dict")
        self._buf.append((key, ev))

    def drain(self) -> List[Dict[str,Any]]:
        out: List[Dict[str,Any]] = []
        for k,e in self._buf:
            out.append(dict(e, key=k))
        self._buf.clear()
        return out

_local = threading.local()


def current() -> _Claims:
    if not hasattr(_local, "ev"):
        _local.ev = _Claims()
    return _local.ev


class ClaimsContext:
    def __init__(self):
        self._claims: List[Dict[str,Any]] = []

    def add_evidence(self, content: bytes | str, meta: Dict[str,Any] | None=None, *,
                     min_trust: float = 0.7) -> Dict[str,Any]:
        if isinstance(content, str):
            content = content.encode("utf-8")
        dg = add_evidence(content, meta or {}, sign=True)
        claim = {"digest": dg, "min_trust": float(min_trust)}
        self._claims.append(claim)
        return claim

    def claims(self) -> List[Dict[str,Any]]:
        return list(self._claims)

    def clear(self) -> None:
        self._claims.clear()

def _current() -> ClaimsContext:
    c = getattr(_ctx_local, "ctx", None)
    if c is None:
        c = ClaimsContext()
        _ctx_local.ctx = c
    return c

def respond_with_evidence(text: str, *,
                          require_hmac: bool=True,
                          min_trust: float=0.7,
                          max_age_s: int | None=None) -> Dict[str,Any]:
    """
    אוכף שקיימות ראיות תקפות בקונטקסט לפני "תשובה".
    מחזיר {"text":..., "claims":[...]} אם עברו Gate.
    """
    claims = _current().claims()
    checked = require(claims, require_hmac=require_hmac, min_trust=min_trust, max_age_s=max_age_s)
    return {"text": text, "claims": checked}

