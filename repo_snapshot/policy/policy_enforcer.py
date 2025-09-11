# policy/policy_enforcer.py

from typing import Dict, Any, Optional, List
from policy.user_policy import UserPolicy
from provenance.signer import verify_hmac
from provenance.castore import ContentAddressableStore, now_s

class PolicyViolation(Exception): ...

class PolicyEnforcer:
    def __init__(self, cas: ContentAddressableStore):
        self.cas = cas

    def enforce_grounding(self, policy: UserPolicy, claims: List[Dict[str, Any]], evidence_records: List[Dict[str, Any]]):
        if not policy.strict_grounding:
            return
        if not claims:
            raise PolicyViolation("Grounding required: missing claims.")
        if not evidence_records:
            raise PolicyViolation("Grounding required: missing evidence records.")

        # Evidence must be signed and fresh
        for ev in evidence_records:
            sig = ev.get("signature")
            key_id = ev.get("key_id")
            digest = ev.get("digest")
            ts = ev.get("timestamp_s")
            if policy.require_signed_evidence and not (sig and key_id and digest):
                raise PolicyViolation("Evidence must be signed with key_id and include digest.")
            if ts is None or (now_s() - ts) > policy.require_freshness_seconds:
                raise PolicyViolation("Evidence expired or missing timestamp.")
            # Verify signature
            if policy.require_signed_evidence and not verify_hmac(ev):
                raise PolicyViolation("Evidence signature invalid.")

            # Verify content exists in CAS and has the same digest
            blob = self.cas.get(digest)
            if blob is None:
                raise PolicyViolation(f"Evidence content not found in CAS for digest {digest}.")

    def enforce_network_host(self, policy: UserPolicy, host: str):
        if not policy.net.is_host_allowed(host):
            raise PolicyViolation(f"Outbound host blocked by policy: {host}")

    def enforce_filesystem(self, policy: UserPolicy, path: str, write: bool, size_bytes: Optional[int] = None):
        if not policy.files.is_path_allowed(path, write):
            raise PolicyViolation(f"File path not allowed or read-only: {path}")
        if size_bytes is not None:
            mb = size_bytes / (1024*1024)
            if mb > policy.files.max_file_mb:
                raise PolicyViolation(f"File too large ({mb:.1f} MB > {policy.files.max_file_mb} MB).")

    def enforce_latency(self, policy: UserPolicy, p95_ms: float):
        if p95_ms > policy.rate.p95_latency_ms_ceiling:
            raise PolicyViolation(f"p95 latency {p95_ms:.0f}ms exceeds ceiling {policy.rate.p95_latency_ms_ceiling}ms.")

    def enforce_capability(self, policy: UserPolicy, capability_name: str):
        if not policy.can_use_capability(capability_name):
            raise PolicyViolation(f"Capability disabled by policy: {capability_name}")