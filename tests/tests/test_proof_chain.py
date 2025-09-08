from __future__ import annotations
import os
from engine.respond_guard import ensure_proof_and_package, RespondBlocked

def test_block_without_claims_when_required(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    try:
        ensure_proof_and_package(response_text="hello", claims=[], policy={"require_claims_for_all_responses": True})
        assert False, "should have blocked"
    except RespondBlocked as rb:
        assert "claims_required" in str(rb)

def test_allow_without_claims_when_not_required(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    out = ensure_proof_and_package(response_text="42", claims=[], policy={"require_claims_for_all_responses": False})
    assert out["ok"] and out["proof_hash"] and out["response_hash"]

def test_inline_evidence_pack_and_store(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    claims = [{
        "id":"c1",
        "text":"Pi is ~3.14159",
        "evidence":[{"kind":"inline","content":"Archimedes approximation"}]
    }]
    out = ensure_proof_and_package(response_text="Pi≈3.14159", claims=claims, policy={"require_claims_for_all_responses": True})
    assert out["ok"]
    assert len(out["proof"]["claims"]) == 1
    assert len(out["proof"]["evidence"]) == 1
    assert out["proof_hash"]

def _fake_http_fetcher_ok(url: str, method: str):
    # מחזיר סטטוס 200 וכותרות סטנדרטיות, ללא גוף
    return (200, {"date":"Tue, 01 Jul 2025 12:00:00 GMT", "content-type":"text/html"}, None)

def _fake_http_fetcher_old(url: str, method: str):
    return (200, {"date":"Tue, 01 Jan 2010 12:00:00 GMT"}, None)

def test_http_evidence_verification(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    claims = [{
        "id":"c2",
        "text":"Official docs available",
        "evidence":[{"kind":"http","url":"https://example.com/docs"}]
    }]
    policy = {"require_claims_for_all_responses": True, "trusted_domains":["example.com"], "max_http_age_days": 3650}
    out = ensure_proof_and_package(response_text="see docs", claims=claims, policy=policy, http_fetcher=_fake_http_fetcher_ok)
    assert out["ok"] and len(out["proof"]["evidence"]) == 1

def test_http_evidence_stale_block(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    claims = [{"id":"c3","text":"old source","evidence":[{"kind":"http","url":"https://example.com/old"}]}]
    policy = {"require_claims_for_all_responses": True, "trusted_domains":["example.com"], "max_http_age_days": 30}
    try:
        ensure_proof_and_package(response_text="uses old source", claims=claims, policy=policy, http_fetcher=_fake_http_fetcher_old)
        assert False, "should block stale"
    except Exception as e:
        assert "evidence_http_stale" in str(e)