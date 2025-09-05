# storage/provenance.py
# -*- coding: utf-8 -*-
import os, json, time, hashlib
from typing import List, Dict
from storage import cas

_BASE = "var/provenance"
os.makedirs(_BASE, exist_ok=True)

def record_provenance(artifact_path: str, sources: List[Dict], trust: float = 0.8) -> str:
    with open(artifact_path, "rb") as f:
        data = f.read()
    art_hash = hashlib.sha256(data).hexdigest()
    entry = {
        "ts": time.time(),
        "artifact_hash": art_hash,
        "artifact_path": artifact_path,
        "sources": sources,
        "trust": trust,
    }
    j = json.dumps(entry, ensure_ascii=False, indent=2)
    h, p = cas.put_text(j)
    return h