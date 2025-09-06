# server/state/ttl.py
from dataclasses import dataclass

@dataclass
class TTLRules:
    # Example policy knobs
    evidence_ttl_s: int = 7 * 24 * 3600
    artifact_ttl_s: int = 14 * 24 * 3600
    per_user_strict: bool = True