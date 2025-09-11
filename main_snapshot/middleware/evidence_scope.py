# imu_repo/middleware/evidence_scope.py
from __future__ import annotations
from grounded.claims import current

def mark_run_start(user_id: str, spec: dict) -> None:
    current().add_evidence("run_start", {
        "source_url": "local://run",
        "trust": 0.95,
        "ttl_s": 3600,
        "payload": {"user_id": user_id, "spec_name": spec.get("name")}
    })