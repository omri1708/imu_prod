# imu_repo/tests/test_trust_and_consistency.py
from __future__ import annotations
import os
from engine.respond_guard import ensure_proof_and_package, RespondBlocked

def _ok_fetch(url: str, method: str):
    return (200, {"date":"Tue, 01 Jul 2025 12:00:00 GMT"}, b"")

def _stale_fetch(url: str, method: str):
    return (200, {"date":"Fri, 01 Jan 2010 12:00:00 GMT"}, b"")

def test_trust_requires_two_distinct_hosts(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    claims = [{
        "id":"latency",
        "text":"p95 latency is 120ms in region X",
        "schema":{"type":"number","unit":"ms","min":0,"max":500,"tolerance":0.1},
        "value": 120,
        "evidence":[
            {"kind":"http","url":"https://a.example.com/latency"},
            {"kind":"http","url":"https://b.example.com/latency"}  # host אחר
        ],
        "consistency_group":"lat-rX"
    }]
    policy = {
        "require_claims_for_all_responses": True,
        "trusted_domains": {"example.com": 3},
        "min_distinct_sources": 2,
        "min_total_trust": 4,  # כל host תורם 3 נק' → סה"כ 6
        "default_number_tolerance": 0.05
    }
    out = ensure_proof_and_package(response_text="latency ok", claims=claims, policy=policy, http_fetcher=_ok_fetch)
    assert out["ok"]

def test_same_host_fails_distinct(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    claims = [{
        "id":"u",
        "text":"uptime 99.9%",
        "schema":{"type":"number","unit":"pct","min":0,"max":100,"tolerance":0.001},
        "value": 99.9,
        "evidence":[
            {"kind":"http","url":"https://example.com/a"},
            {"kind":"http","url":"https://example.com/b"}  # אותו host
        ],
        "consistency_group":"uptime"
    }]
    policy = {
        "require_claims_for_all_responses": True,
        "trusted_domains": {"example.com": 2},
        "min_distinct_sources": 2,
        "min_total_trust": 2
    }
    try:
        ensure_proof_and_package(response_text="ok", claims=claims, policy=policy, http_fetcher=_ok_fetch)
        assert False, "should have failed distinct sources"
    except RespondBlocked as e:
        assert "distinct sources" in str(e)

def test_schema_range_block(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    claims = [{
        "id":"too_big",
        "text":"latency 9999ms",
        "schema":{"type":"number","unit":"ms","min":0,"max":500},
        "value": 9999,
        "evidence":[{"kind":"inline","content":"logs ..."}]
    }]
    policy = {"require_claims_for_all_responses": True, "inline_trust": 1, "min_total_trust": 1}
    try:
        ensure_proof_and_package(response_text="nope", claims=claims, policy=policy)
        assert False, "should block due to schema max"
    except RespondBlocked as e:
        assert "max" in str(e).lower()

def test_consistency_group_numeric_tol(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    claims = [
        {
            "id":"a",
            "text":"p95 100ms",
            "schema":{"type":"number","unit":"ms","min":0,"max":500,"tolerance":0.2},
            "value": 100,
            "evidence":[{"kind":"inline","content":"calc1"}],
            "consistency_group":"g1"
        },
        {
            "id":"b",
            "text":"p95 115ms",
            "schema":{"type":"number","unit":"ms"},
            "value": 115,
            "evidence":[{"kind":"inline","content":"calc2"}],
            "consistency_group":"g1"
        }
    ]
    policy = {
        "require_claims_for_all_responses": True,
        "inline_trust": 1,
        "min_total_trust": 1,
        "require_consistency_groups": True,
        "default_number_tolerance": 0.2
    }
    out = ensure_proof_and_package(response_text="ok", claims=claims, policy=policy)
    assert out["ok"]

def test_consistency_group_block_on_large_diff(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    claims = [
        {
            "id":"a",
            "text":"p95 100ms",
            "schema":{"type":"number","unit":"ms","tolerance":0.05},
            "value": 100,
            "evidence":[{"kind":"inline","content":"calc1"}],
            "consistency_group":"g2"
        },
        {
            "id":"b",
            "text":"p95 140ms",
            "schema":{"type":"number","unit":"ms"},
            "value": 140,
            "evidence":[{"kind":"inline","content":"calc2"}],
            "consistency_group":"g2"
        }
    ]
    policy = {
        "require_claims_for_all_responses": True,
        "inline_trust": 1,
        "min_total_trust": 1,
        "require_consistency_groups": True,
        "default_number_tolerance": 0.1
    }
    try:
        ensure_proof_and_package(response_text="ok", claims=claims, policy=policy)
        assert False, "should block due to inconsistency"
    except RespondBlocked as e:
        assert "inconsistent values" in str(e)