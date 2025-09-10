# imu_repo/engine/synthesis_pipeline.py
from __future__ import annotations
from typing import Any, Dict, List
import time
from policy.user_policy import UserPolicy
from policy.policy_enforcer import PolicyEnforcer
from engine.grounding_gate import require_grounded_response
from provenance.castore import ContentAddressableStore


class SynthesisPipeline:
    def __init__(self, cas_dir: str):
        self.cas = ContentAddressableStore(cas_dir)
        self.enforcer = PolicyEnforcer(self.cas)

    def run(self,
            user_policy: UserPolicy,
            plan: Dict[str, Any],
            generated: Dict[str, Any],
            tests_result: Dict[str, Any],
            verification: Dict[str, Any]) -> Dict[str, Any]:
        """
        מחבר Plan→Generate→Test→Verify→Package ומחזיר תגובה *רק* עם ראיות חתומות ותקפות.
        """
        start = time.time()

        # claims & evidence מהשלבים הקודמים:
        claims: List[Dict[str, Any]] = verification.get("claims", [])
        raw_evidence: List[Dict[str, Any]] = verification.get("evidence", [])

        # הכנסה ל־CAS + חתימה
        evidence_records: List[Dict[str, Any]] = []
        for ev in raw_evidence:
            blob = ev["content"].encode("utf-8") if isinstance(ev["content"], str) else ev["content"]
            digest = self.cas.put(blob)
            from ..provenance.signer import signed_evidence
            evidence_records.append(signed_evidence(
                digest=digest, source=ev.get("source", "unknown"), trust=ev.get("trust", "low"), metadata=ev.get("meta", {})
            ))

        # אכיפת Grounding ו־Policy לפני תשובה
        require_grounded_response(user_policy, self.enforcer, claims, evidence_records)

        # מדידת p95 (פשטני: כאן משך ריצה; בפועל יש לאסוף דגימות)
        dur_ms = (time.time()-start)*1000
        self.enforcer.enforce_latency(user_policy, p95_ms=dur_ms)

        # חבילת תשובה
        return {
            "ok": True,
            "plan": plan,
            "generated": generated,
            "tests": tests_result,
            "verification": {"claims": claims, "evidence": evidence_records, "p95_ms": dur_ms}
        }

