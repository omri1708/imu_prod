# imu_repo/engine/verifier.py
from __future__ import annotations
from typing import Dict, Any, List
from engine.cas_verify import verify_bundle_signature
from engine.consistency import validate_claims_and_consistency, ConsistencyError
from engine.provenance import enforce_min_provenance, ProvenanceError
from engine.trust_tiers import enforce_trust_requirements, TrustPolicyError
from engine.key_delegation import expand_keyring_with_chain, DelegationError, enforce_scope_for_kid
from engine.evidence_freshness import enforce_claims_freshness, FreshnessError 

class VerificationFailed(Exception): ...

def verify_bundle(bundle: Dict[str,Any], policy: Dict[str,Any], *, keyring: Dict[str,Dict[str,str]], http_fetcher=None) -> Dict[str,Any]:
    if not verify_bundle_signature(bundle, keyring):
        raise VerificationFailed("CAS signature invalid")
    claims = bundle.get("claims") or []
    if not isinstance(claims, list):
        raise VerificationFailed("invalid claims array")
    try:
        validate_claims_and_consistency(
            claims,
            require_consistency_groups=bool(policy.get("require_consistency_groups", False)),
            default_number_tolerance=float(policy.get("default_number_tolerance", 0.01))
        )
        for c in claims:
            enforce_trust_requirements(c, policy)
            enforce_min_provenance(c, policy, http_fetcher=http_fetcher)
        # בדיקת טריות לכל ה-claims (ייתכן SLA פר-סוג/ברירת־מחדל)
        enforce_claims_freshness(claims, policy)
    except (TrustPolicyError, ProvenanceError, ConsistencyError, FreshnessError) as e:
        raise VerificationFailed(str(e))
    return {"ok": True, "claims": len(claims)}

def verify_bundle_with_chain(bundle: Dict[str,Any], policy: Dict[str,Any], *, root_keyring: Dict[str,Dict[str,str]], trust_chain: List[Dict[str,Any]], http_fetcher=None, expected_scope: str | None=None) -> Dict[str,Any]:
    try:
        kr = expand_keyring_with_chain(root_keyring, trust_chain)
    except DelegationError as e:
        raise VerificationFailed(f"delegation error: {e}")
    if expected_scope:
        sig = bundle.get("signature") or {}
        kid = sig.get("key_id")
        if not kid:
            raise VerificationFailed("missing key_id in signature")
        try:
            enforce_scope_for_kid(trust_chain, kid, expected_scope)
        except DelegationError as e:
            raise VerificationFailed(f"scope error: {e}")
    return verify_bundle(bundle, policy, keyring=kr, http_fetcher=http_fetcher)

# עטיפה לשימוש ב-quorum_verify
def as_quorum_member(keyring: Dict[str,Dict[str,str]], *, http_fetcher=None):
    def _fn(bundle: Dict[str,Any], policy: Dict[str,Any]) -> Dict[str,Any]:
        try:
            out = verify_bundle(bundle, policy, keyring=keyring, http_fetcher=http_fetcher)
            return {"ok": True, **out}
        except Exception as e:
            return {"ok": False, "reason": str(e)}
    return _fn

def as_quorum_member_with_chain(root_keyring: Dict[str,Dict[str,str]], trust_chain: List[Dict[str,Any]], *, http_fetcher=None, expected_scope: str | None=None):
    def _fn(bundle: Dict[str,Any], policy: Dict[str,Any]) -> Dict[str,Any]:
        try:
            out = verify_bundle_with_chain(bundle, policy, root_keyring=root_keyring, trust_chain=trust_chain, http_fetcher=http_fetcher, expected_scope=expected_scope)
            return {"ok": True, **out}
        except Exception as e:
            return {"ok": False, "reason": str(e)}
    return _fn