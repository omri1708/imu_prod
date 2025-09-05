# imu_repo/engine/verifier.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
from engine.cas_verify import verify_bundle_signature, CasVerifyError
from engine.consistency import validate_claims_and_consistency, ConsistencyError
from engine.provenance import enforce_min_provenance, ProvenanceError
from engine.trust_tiers import enforce_trust_requirements, TrustPolicyError

class VerificationFailed(Exception): ...

def verify_bundle(bundle: Dict[str,Any], policy: Dict[str,Any], *, keyring: Dict[str,Dict[str,str]], http_fetcher=None) -> Dict[str,Any]:
    """
    מאמת:
      1) חתימה
      2) מבנה טענות
      3) Trust / Provenance
      4) עקביות מספרית/קבוצתית (לוקלית)
      5) רענון evidences L3 (אם צריך)
    """
    # 1) חתימה
    if not verify_bundle_signature(bundle, keyring):
        raise VerificationFailed("CAS signature invalid")

    claims = bundle.get("claims") or []
    if not isinstance(claims, list):
        raise VerificationFailed("invalid claims array")

    # 2/3/4
    try:
        validate_claims_and_consistency(
            claims,
            require_consistency_groups=bool(policy.get("require_consistency_groups", False)),
            default_number_tolerance=float(policy.get("default_number_tolerance", 0.01))
        )
        for c in claims:
            enforce_trust_requirements(c, policy)
            enforce_min_provenance(c, policy, http_fetcher=http_fetcher)  # יבצע fetch אם צריך
    except (TrustPolicyError, ProvenanceError, ConsistencyError) as e:
        raise VerificationFailed(str(e))

    return {"ok": True, "claims": len(claims)}