# tests/test_policy_and_provenance.py
import os, json, tempfile
import time, base64

from imu.policy.user_policy import UserPolicy
from imu.policy.policy_enforcer import PolicyEnforcer, PolicyViolation
from imu.provenance.castore import ContentAddressableStore
from imu.provenance.signer import signed_evidence
from policy.policy_rules import POLICY
from governance.enforcement import assert_net, assert_path, ratelimit, enforce_p95, hard_ground, attach_and_sign_evidence
from provenance.signer import record_evidence, verify_signature, save_json, load_json, get_blob

def test_net_allowlist_ok():
    uid = "u1"
    assert POLICY.allow_url(uid, "https://api.github.com/repos"), "github should be allowed"

def test_net_block_http():
    uid = "u1"
    assert not POLICY.allow_url(uid, "http://example.com"), "plain http is blocked"

def test_file_whitelist():
    uid = "u1"
    os.makedirs("./workspace/tmp", exist_ok=True)
    assert POLICY.allow_path(uid, "./workspace/tmp/file.txt", True)
    assert not POLICY.allow_path(uid, "/etc/passwd", False)

def test_rate_limit():
    uid = "u1"
    ok = 0
    for _ in range(5):
        if POLICY.rate_allow(uid, "ui_push", 1):
            ok += 1
    assert ok >= 1

def test_p95_guard():
    uid = "u1"
    enforce_p95(uid, "plan", 1000)  # within default 1500

def test_provenance_sign_and_verify():
    uid = "u2"
    ev = record_evidence(uid, {"kind": "doc", "value": {"answer": 42}, "trust": 0.7}, 0.7)
    assert verify_signature(uid, ev["sig_id"])
    digest = ev["digest"]
    payload = load_json(digest)
    assert payload["value"]["answer"] == 42

def test_hard_ground_enforces_evidence():
    uid = "u3"
    claim = {"type": "fact", "value": "foo"}
    # בלי ראיה—חייב fail
    failed = False
    try:
        hard_ground(uid, [claim])
    except ValueError:
        failed = True
    assert failed

    claim2 = {"type": "fact", "value": "bar"}
    claim2 = attach_and_sign_evidence(uid, claim2, {"kind":"doc","value":{"bar":1},"trust":0.6})
    ok = hard_ground(uid, [claim2])
    assert ok and ok[0]["evidence"]["digest"]

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