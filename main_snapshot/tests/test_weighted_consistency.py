# imu_repo/tests/test_weighted_consistency.py
from __future__ import annotations
from engine.respond_guard import ensure_proof_and_package, RespondBlocked

def _fetch_ok(url: str, method: str):
    return (200, {"date":"Tue, 01 Jul 2025 12:00:00 GMT"}, b"")

def test_weighted_dominates_allows_conflict(tmp_path, monkeypatch):
    import os, json
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    pol = {
        "trust_domains": {"example.com":3},
        "trusted_domains": ["example.com"],
        "min_distinct_sources": 1,
        "min_total_trust": 1,
        "min_provenance_level": 1,
        "global_consistency": {
            "relations":[{"a":"A:x","b":"B:x","rel":"equal","tol_pct":0.0,"dominates":"A:x"}],
            "weights":{"A:x":5.0, "B:x":1.0}
        }
    }
    claims = [
        {"id":"A:x","text":"A=100","schema":{"type":"number"},"value":100,"evidence":[{"kind":"http","url":"https://example.com/a"}]},
        {"id":"B:x","text":"B=101","schema":{"type":"number"},"value":101,"evidence":[{"kind":"http","url":"https://example.com/b"}]},
    ]
    # אמור לעבור בזכות דומיננטיות A:x
    out = ensure_proof_and_package(response_text="ok", claims=claims, policy=pol, http_fetcher=_fetch_ok)
    assert out["ok"]

def test_weighted_conflict_blocks(tmp_path):
    import os
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    pol = {
        "min_distinct_sources": 1,
        "min_total_trust": 0,
        "min_provenance_level": 0,
        "global_consistency": {
            "relations":[{"a":"A:y","b":"B:y","rel":"equal","tol_pct":0.0}],
            "weights":{"A:y":1.0,"B:y":1.0}
        }
    }
    claims = [
        {"id":"A:y","text":"A=10","schema":{"type":"number"},"value":10,"evidence":[{"kind":"inline"}]},
        {"id":"B:y","text":"B=12","schema":{"type":"number"},"value":12,"evidence":[{"kind":"inline"}]},
    ]
    try:
        ensure_proof_and_package(response_text="x", claims=claims, policy=pol, http_fetcher=lambda u,m:(200,{},b""))
        assert False, "should block"
    except RespondBlocked as e:
        assert "equal conflict" in str(e).lower()