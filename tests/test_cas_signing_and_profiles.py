# imu_repo/tests/test_cas_signing_and_profiles.py
from __future__ import annotations
import os, json
from engine.policy_compiler import compile_with_profiles, policy_passes
from engine.respond_guard import ensure_proof_and_package

def _fetch_ok(url: str, method: str):
    return (200, {"date":"Tue, 01 Jul 2025 12:00:00 GMT"}, b"")

def test_profiles_and_cas_signature(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    base = {
        "trust_domains": {"example.com":3, "acme.org":2},
        "trusted_domains": ["example.com","acme.org"],
        "min_distinct_sources": 1,
        "min_total_trust": 2,
        "signing_keys": {"main":{"secret_hex":"ab"*32, "algo":"sha256"}},
        "min_provenance_level": 2,
        "p95_limits": {"plan": 100}
    }
    profs = compile_with_profiles(json.dumps(base))
    pol = policy_passes(profs["stage"])

    claims = [{
        "id":"lat:p95",
        "type":"latency",
        "text":"p95=80ms",
        "schema":{"type":"number","unit":"ms","min":0,"max":500},
        "value": 80,
        "evidence":[{"kind":"http","url":"https://api.example.com/metrics"}],
        "consistency_group":"lat"
    }]

    out = ensure_proof_and_package(response_text="ok", claims=claims, policy=pol, http_fetcher=_fetch_ok)
    assert out["ok"]
    bundle = out["proof"]
    assert "signature" in bundle and "sig" in bundle["signature"]