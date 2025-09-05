# tests/test_multi_user_policy.py
# -*- coding: utf-8 -*-
import time
from governance.user_policy import ensure_user, get_user_policy, restrict_domains, raise_trust_floor
from engine.contracts_gate import enforce_respond_contract, PolicyDenied
from grounded.evidence_contracts import EvidenceIndex

def test_policy_denies_wrong_domain():
    user = "strict_org"
    ensure_user(user)
    pol, ev = get_user_policy(user)
    restrict_domains(user, ["corp.example"])  # אין example.com
    raise_trust_floor(user, 0.95)

    # עדות "טובה" אבל מדומיין אסור
    sha = "a"*64
    ev.put(sha, {"sha256":sha,"ts":int(time.time()),"trust":0.99,"url":"https://example.com/a","sig_ok":True})

    try:
        enforce_respond_contract("pipeline_generate",
                                 [{"id":"c1","text":"ok"}],
                                 [{"sha256":sha,"ts":int(time.time()),"trust":0.99,"url":"https://example.com/a","sig_ok":True}],
                                 pol, ev)
    except PolicyDenied as e:
        assert "evidence_invalid" in str(e)
    else:
        raise AssertionError("expected PolicyDenied")

def test_policy_allows_corp_domain():
    user = "strict_org"
    ensure_user(user)
    pol, ev = get_user_policy(user)
    restrict_domains(user, ["corp.example"])
    sha = "b"*64
    ev.put(sha, {"sha256":sha,"ts":int(time.time()),"trust":0.99,"url":"https://corp.example/x","sig_ok":True})

    enforce_respond_contract("answer",
                             [{"id":"c","text":"ok"}],
                             [{"sha256":sha,"ts":int(time.time()),"trust":0.99,"url":"https://corp.example/x","sig_ok":True}],
                             pol, ev)  # לא יזרוק