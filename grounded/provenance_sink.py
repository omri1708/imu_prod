# imu_repo/grounded/provenance_sink.py
from __future__ import annotations
from typing import Optional, Dict, Any
from provenance.cas import CAS
from provenance.provenance import ProvenanceStore
from grounded.claims import current

def flush_current_evidences_to_cas(cas_root: str, *, min_trust: float=0.75) -> Dict[str,Any]:
    """
    מושך את כל הראיות שנאספו ב-session הנוכחי, שומר ב-CAS,
    ומחזיר מפתח זיהוי (sha של מסמך הראיות).
    """
    cas = CAS(cas_root)
    store = ProvenanceStore(cas, min_trust=min_trust)
    ev_sha = store.ingest_evidences(current().snapshot())
    return {"evidences_sha": ev_sha, "min_trust": min_trust}