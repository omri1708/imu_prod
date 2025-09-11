# tests/test_contracts_strict.py
# -*- coding: utf-8 -*-
import pytest, time
from governance.policy import RespondPolicy, EvidenceRule
from grounded.evidence_contracts import EvidenceIndex
from engine.contracts_gate import enforce_respond_contract

def _mk_ev(idx: EvidenceIndex, trust=0.9, age=0, sig_ok=True, url="https://example.com/doc"):
    ts = int(time.time()) - age
    sha = "0"*64  # נזין ב-idx ממילא
    idx.put(sha, {"ts": ts, "trust":trust, "url":url, "sig_ok":sig_ok})
    return {"sha256": sha, "ts": ts, "trust": trust, "url": url, "sig_ok": sig_ok}

def test_require_evidence_and_claims_ok():
    pol = RespondPolicy(require_claims=True, require_evidence=True,
                        evidence=EvidenceRule(min_trust=0.7, max_age_sec=60, allowed_domains=["example.com"], require_signature=True))
    idx = EvidenceIndex()
    claims = [{"id":"c1","text":"2+2=4"}]
    ev = [_mk_ev(idx, trust=0.9, age=1, sig_ok=True, url="https://example.com/x")]
    enforce_respond_contract("answer", claims, ev, pol, idx)

def test_evidence_trust_too_low():
    pol = RespondPolicy(require_claims=True, require_evidence=True,
                        evidence=EvidenceRule(min_trust=0.8, max_age_sec=60, allowed_domains=[], require_signature=True))
    idx = EvidenceIndex()
    claims = [{"id":"c1","text":"x"}]
    ev = [_mk_ev(idx, trust=0.5)]
    with pytest.raises(Exception):
        enforce_respond_contract("answer", claims, ev, pol, idx)

def test_evidence_domain_forbidden():
    pol = RespondPolicy(require_claims=True, require_evidence=True,
                        evidence=EvidenceRule(min_trust=0.1, max_age_sec=60, allowed_domains=["trusted.org"], require_signature=False))
    idx = EvidenceIndex()
    claims = [{"id":"c1","text":"x"}]
    ev = [_mk_ev(idx, url="https://evil.com/doc")]
    with pytest.raises(Exception):
        enforce_respond_contract("answer", claims, ev, pol, idx)