# imu_repo/tests/test_respond_integration.py
from __future__ import annotations
import os
from engine.pipeline_respond_hook import pipeline_respond

def test_block_without_claims(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    ctx = {"__policy__": {"require_claims_for_all_responses": True}}
    out = pipeline_respond(ctx=ctx, answer_text="hello world")
    assert not out["ok"]
    assert "claims_required" in out["error"]

def test_pass_with_inline_claims(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    ctx = {
        "__policy__": {"require_claims_for_all_responses": True},
        "__claims__": [{
            "text": "the sky appears blue",
            "evidence": [{"kind":"inline","content":"Rayleigh scattering in atmosphere"}]
        }]
    }
    out = pipeline_respond(ctx=ctx, answer_text="The sky looks blue.")
    assert out["ok"]
    assert out["proof_hash"] and out["response_hash"]

def _fake_http_fetcher_ok(url: str, method: str):
    return (200, {"date":"Tue, 01 Jul 2025 12:00:00 GMT", "content-type":"text/html"}, b"")

def test_pass_with_http_claims_via_ctx_fetcher(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    ctx = {
        "__policy__": {
            "require_claims_for_all_responses": True,
            "trusted_domains": ["example.com"],
            "http_download_for_hash": False  # HEAD מספיק; שומר דטרמיניזם
        },
        "__http_fetcher__": _fake_http_fetcher_ok,
        "__claims__": [{
            "text":"public docs exist",
            "evidence":[{"kind":"http","url":"https://example.com/docs"}]
        }]
    }
    out = pipeline_respond(ctx=ctx, answer_text="See the docs.")
    assert out["ok"], f"unexpected block: {out}"
    proof = out["proof"]
    assert len(proof["evidence"]) == 1
    assert proof["evidence"][0]["kind"] == "http"

def _fake_http_fetcher_old(url: str, method: str):
    return (200, {"date":"Tue, 01 Jan 2010 12:00:00 GMT"}, b"")

def test_stale_http_blocked_via_ctx_policy(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    ctx = {
        "__policy__": {
            "require_claims_for_all_responses": True,
            "trusted_domains": ["example.com"],
            "max_http_age_days": 30
        },
        "__http_fetcher__": _fake_http_fetcher_old,
        "__claims__": [{
            "text":"old link",
            "evidence":[{"kind":"http","url":"https://example.com/old"}]
        }]
    }
    out = pipeline_respond(ctx=ctx, answer_text="uses old link")
    assert not out["ok"]
    assert "evidence_http_stale" in out["error"]