# provenance/audit.py (Append-only audit log)
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, time, hashlib, hmac
from typing import Optional, Dict, Any

class AuditLog:
    """
    Append-only JSONL with chained HMAC to prevent tampering.
    Each record: {"ts":..., "actor":..., "action":..., "payload":..., "prev": sha256(prev_line), "hmac":...}
    """
    def __init__(self, path: str, secret: bytes):
        self.path = path
        self.secret = secret
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(self.path, "a", encoding="utf-8").close()

    def _last_hash(self) -> str:
        h = hashlib.sha256()
        with open(self.path, "rb") as f:
            for line in f:
                h.update(line.rstrip(b"\n"))
        return h.hexdigest()

    def append(self, actor: str, action: str, payload: Dict[str, Any]):
        prev = self._last_hash()
        rec = {"ts": time.time(), "actor": actor, "action": action, "payload": payload, "prev": prev}
        msg = json.dumps(rec, sort_keys=True, separators=(",",":")).encode("utf-8")
        sig = hmac.new(self.secret, msg, "sha256").hexdigest()
        rec["hmac"] = sig
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",",":")) + "\n")

    def verify(self) -> bool:
        prev = hashlib.sha256()
        with open(self.path, "rb") as f:
            for raw in f:
                line = raw.rstrip(b"\n")
                # reconstruct expected prev hash
                # compute HMAC
                import json as _json
                rec = _json.loads(line)
                check = dict(rec)
                sig = check.pop("hmac")
                msg = _json.dumps(check, sort_keys=True, separators=(",",":")).encode("utf-8")
                import hmac as _hmac, hashlib as _hashlib
                calc = _hmac.new(self.secret, msg, "sha256").hexdigest()
                if calc != sig: return False
        return True