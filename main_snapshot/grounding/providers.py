# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, time, hashlib
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from assurance.cas import CAS, sha256_bytes
from assurance.errors import ResourceRequired

@dataclass
class EvidenceRecord:
    kind: str
    source: str
    digest: str
    ts: float
    trust: float

class FileProvider:
    def __init__(self, cas: CAS): self.cas = cas
    def fetch(self, path: str, min_mtime: float = 0, trust: float = 0.6) -> EvidenceRecord:
        st = os.stat(path)
        if st.st_mtime < min_mtime:
            raise ValueError("stale file")
        d = self.cas.put_file(path, meta={"provider":"file"})
        return EvidenceRecord("file", os.path.abspath(path), d, time.time(), trust)

class HTTPProvider:
    def __init__(self, cas: CAS): self.cas = cas
    def fetch(self, url: str, min_age_sec: int = 0, trust: float = 0.5) -> EvidenceRecord:
        try:
            import requests
        except Exception:
            raise ResourceRequired("tool:requests", "pip install requests")
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        # freshness header (optional)
        d = self.cas.put_bytes(r.content, meta={"provider":"http", "url": url, "headers": dict(r.headers)})
        return EvidenceRecord("http", url, d, time.time(), trust)

def add_evidence_to_session(session, ev: EvidenceRecord, signer=None):
    meta = {"kind": ev.kind, "source": ev.source, "digest": ev.digest, "ts": ev.ts, "trust": ev.trust}
    if signer:
        sig = signer.sign(json.dumps(meta, sort_keys=True).encode("utf-8"))
        meta["signature"] = sig
    session.add_evidence(ev.kind, ev.source, ev.digest, ev.trust, ev.ts, signed=meta.get("signature"))
