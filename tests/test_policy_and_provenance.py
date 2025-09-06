# tests/test_policy_and_provenance.py
import os, json, tempfile
from imu.policy.user_policy import UserPolicy
from imu.policy.policy_enforcer import PolicyEnforcer, PolicyViolation
from imu.provenance.castore import ContentAddressableStore
from imu.provenance.signer import signed_evidence

def test_signed_evidence_and_policy_ok():
    with tempfile.TemporaryDirectory() as d:
        cas = ContentAddressableStore(d)
        enforcer = PolicyEnforcer(cas)
        pol = UserPolicy(user_id="u1", trust="high")

        digest = cas.put(b"hello-evidence")
        ev = signed_evidence(digest, "unit-test", "high", {"k":"v"})
        claims = [{"type":"fact","value":"demo"}]
        enforcer.enforce_grounding(pol, claims, [ev])  # לא יזרוק

def test_missing_evidence_violates():
    with tempfile.TemporaryDirectory() as d:
        cas = ContentAddressableStore(d)
        enforcer = PolicyEnforcer(cas)
        pol = UserPolicy(user_id="u1", trust="low")
        try:
            enforcer.enforce_grounding(pol, [{"t":"x"}], [])
            assert False, "should have raised"
        except PolicyViolation:
            pass