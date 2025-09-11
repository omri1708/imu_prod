# imu_repo/engine/verifier_km.py
from __future__ import annotations
from typing import Dict, Any, Callable
from engine.keychain_manager import KeychainManager
from engine.verifier import verify_bundle_with_chain

def as_quorum_member_with_km(km: KeychainManager, *, http_fetcher=None, expected_scope: str | None=None):
    def _fn(bundle: Dict[str,Any], policy: Dict[str,Any]) -> Dict[str,Any]:
        try:
            chain, kr = km.current()
            out = verify_bundle_with_chain(bundle, policy, root_keyring=kr, trust_chain=chain, http_fetcher=http_fetcher, expected_scope=expected_scope)
            return {"ok": True, **out}
        except Exception as e:
            return {"ok": False, "reason": str(e)}
    return _fn