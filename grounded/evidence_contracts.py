# grounded/evidence_contracts.py
# -*- coding: utf-8 -*-
import hashlib, json, time
from dataclasses import dataclass
from typing import Optional, Dict, Any
from contracts.errors import ContractViolation

@dataclass
class Evidence:
    sha256: str
    ts: int
    trust: float
    url: str
    sig_ok: bool

def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

class EvidenceIndex:
    """אינדקס ראיות: שמירה/שליפה + אימות התאמה (sha/timestamp/trust)."""
    def __init__(self):
        self._idx: Dict[str, Dict[str, Any]] = {}

    def put(self, sha256: str, meta: Dict[str, Any]):
        self._idx[sha256] = {
            "ts": int(meta.get("ts", int(time.time()))),
            "trust": float(meta.get("trust", 0.5)),
            "url": str(meta.get("url", "")),
            "sig_ok": bool(meta.get("sig_ok", False)),
        }

    def get(self, sha256: str) -> Optional[Dict[str, Any]]:
        return self._idx.get(sha256)

    def verify(self, ev: Evidence) -> None:
        rec = self.get(ev.sha256)
        if not rec:
            raise ContractViolation("evidence_not_found", detail={"sha256": ev.sha256})
        if int(rec["ts"]) != int(ev.ts):
            raise ContractViolation("evidence_ts_mismatch", detail={"sha256": ev.sha256})
        if abs(float(rec["trust"]) - float(ev.trust)) > 1e-9:
            raise ContractViolation("evidence_trust_mismatch", detail={"sha256": ev.sha256})
        if bool(rec["sig_ok"]) != bool(ev.sig_ok):
            raise ContractViolation("evidence_sig_mismatch", detail={"sha256": ev.sha256})
        if rec["url"] != ev.url:
            raise ContractViolation("evidence_url_mismatch", detail={"sha256": ev.sha256})

    @staticmethod
    def serialize(ev: Evidence) -> str:
        return json.dumps(ev.__dict__, separators=(",",":"), ensure_ascii=False)