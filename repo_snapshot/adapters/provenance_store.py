# adapters/provenance_store.py
# -*- coding: utf-8 -*-
import os, hashlib, time, json
from typing import Dict, Any
from grounded.evidence_contracts import EvidenceIndex

CAS_DIR = "var/cas"

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def cas_put(filename: str, content: bytes) -> str:
    os.makedirs(CAS_DIR, exist_ok=True)
    sha = _sha256_bytes(content)
    path = os.path.join(CAS_DIR, sha)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(content)
    # קישור סימבולי בשם ידידותי
    alias = os.path.join(CAS_DIR, filename.replace("/", "__"))
    try:
        if os.path.islink(alias) or os.path.exists(alias):
            os.remove(alias)
        os.symlink(sha, alias)
    except Exception:
        # סביבות בלי symlink
        with open(alias+".json","w",encoding="utf-8") as f:
            json.dump({"sha256":sha,"ts":int(time.time())}, f)
    return sha

def evidence_for(sha: str, *, domain: str = "cas.local", trust: float = 0.99) -> Dict[str,Any]:
    return {"sha256":sha,"ts":int(time.time()),"trust":trust,"url":f"https://{domain}/{sha}","sig_ok":True}

def register_evidence(ev_index: EvidenceIndex, ev: Dict[str,Any]) -> None:
    ev_index.put(ev["sha256"], ev)