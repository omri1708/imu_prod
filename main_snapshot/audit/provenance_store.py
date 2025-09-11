# imu_repo/audit/provenance_store.py
from __future__ import annotations
from typing import Dict, Any
import time
from audit.cas import put_json
from audit.ledger import append

def record_evidence(kind: str, evidence: Dict[str,Any], *, actor: str, obj: str, tags: list[str] | None=None) -> Dict[str,Any]:
    """
    - שומר ראיה ב-CAS (sha256)
    - רושם תוך שרשור־האשים ב-ledger (tamper-evident)
    """
    cas = put_json({"kind":kind, "evidence": evidence}, meta={"kind": kind})
    ev_id = f"{kind}:{cas['sha256']}"
    led = append({
        "actor": actor,
        "action":"evidence.put",
        "object": obj,
        "evidence_id": ev_id,
        "sha256": cas["sha256"],
        "tags": tags or [],
    })
    return {"evidence_id": ev_id, "sha256": cas["sha256"], "ledger": led}