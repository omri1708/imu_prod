# storage/provenance_store.py
from __future__ import annotations
import hashlib, json, time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Literal, Any

Trust = Literal["unknown","low","medium","high","pinned"]

@dataclass
class Evidence:
    content_sha256: str
    uri: Optional[str]
    fetched_at: float
    trust: Trust
    signed_by: Optional[str] = None  # key id
    signature: Optional[str] = None  # hex
    schema: Optional[str] = None     # JSON schema uri/name
    meta: Dict[str, Any] = None

class ProvenanceStore:
    """
    Content-addressable store for UI/artifacts/claims.
    Files/bytes hashed; metadata carries trust & optional signature.
    """
    def __init__(self):
        self._objects: Dict[str, bytes] = {}
        self._evidence: Dict[str, Evidence] = {}

    @staticmethod
    def sha256_bytes(b: bytes) -> str:
        return hashlib.sha256(b).hexdigest()

    def put_bytes(self, b: bytes, uri: Optional[str], trust: Trust, signed_by=None, signature=None, schema=None, meta=None) -> str:
        h = self.sha256_bytes(b)
        self._objects[h] = b
        self._evidence[h] = Evidence(
            content_sha256=h, uri=uri, fetched_at=time.time(),
            trust=trust, signed_by=signed_by, signature=signature, schema=schema, meta=meta or {}
        )
        return h

    def get_bytes(self, sha: str) -> bytes:
        return self._objects[sha]

    def get_evidence(self, sha: str) -> Evidence:
        return self._evidence[sha]

    def export_manifest(self) -> bytes:
        man = {sha: asdict(ev) for sha, ev in self._evidence.items()}
        return json.dumps(man, ensure_ascii=False, indent=2).encode("utf-8")

# singleton
prov = ProvenanceStore()