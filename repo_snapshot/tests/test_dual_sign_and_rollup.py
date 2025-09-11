# imu_repo/tests/test_dual_sign_and_rollup.py
from __future__ import annotations
import os, json, time, tempfile
from engine.policy_compiler import strict_prod_from, keyring_from_policy
from engine.respond_guard import ensure_proof_and_package
from engine.verifier import verify_bundle
from engine.audit_rollup import rollup_window

def _http_ok(url: str, method: str):
    # מחזיר תאריך "חדש" עבור L3
    return (200, {"date":"Tue, 01 Jul 2025 12:00:00 GMT"}, b"{}")

def test_producer_signs_and_verifier_checks(tmp_path, monkeypatch):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    base = {
        "trust_domains": {"example.com":4, "acme.org":3},
        "trusted_domains": ["example.com","acme.org"],
        "signing_keys": {"main":{"secret_hex":"ab"*32, "algo":"sha256"}},
        "min_distinct_sources": 1,  # יוגדל ב-strict
        "min_total_trust": 1,
        "min_provenance_level": 2,
        "p95_limits": {"plan": 200}
    }
    pol = strict_prod_from(json.dumps(base))
    kr = keyring_from_policy(pol)

    claims = [{
        "id":"lat:p95",
        "type":"latency",
        "text":"p95 is 120ms",
        "schema":{"type":"number","unit":"ms","min":0,"max":1000},
        "value":120,
        "evidence":[{"kind":"http","url":"https://api.example.com/metrics"}],
        "consistency_group":"lat"
    },{
        "id":"ver:source",
        "type":"meta",
        "text":"source=example.com",
        "schema":{"type":"string"},
        "value":"example.com",
        "evidence":[{"kind":"inline"}],
        "consistency_group":"meta"
    }]

    produced = ensure_proof_and_package(response_text="OK", claims=claims, policy=pol, http_fetcher=_http_ok)
    assert produced["ok"]
    bundle = produced["proof"]

    verified = verify_bundle(bundle, pol, keyring=kr, http_fetcher=_http_ok)
    assert verified["ok"]

def test_audit_rollup_signed(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    # צור כמה אירועים בדויים
    d = os.environ["IMU_AUDIT_DIR"]
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "events.jsonl")
    now = time.time()
    with open(path, "w", encoding="utf-8") as f:
        for i in range(5):
            obj = {"ts": now - (i*10), "evt": "x", "n": i}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    signing_key = {"roll":{"secret_hex":"cd"*32, "algo":"sha256"}}
    out = rollup_window(window_seconds=3600, signing_key=signing_key)
    assert out["ok"]
    assert "signature" in out["rollup"]
    assert out["rollup"]["count"] == 5
    assert isinstance(out["rollup"]["root"], str) and len(out["rollup"]["root"]) == 64