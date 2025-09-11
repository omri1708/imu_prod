# imu_repo/tests/test_stage100_verify_and_nearmiss.py
from __future__ import annotations
import os, json, time
from typing import List, Dict, Any
from engine.policy_compiler import strict_prod_from, keyring_from_policy
from engine.respond_guard import ensure_proof_and_package
from engine.key_delegation import issue_delegation, derive_child_secret_hex
from engine.keychain_manager import KeychainManager
from engine.verifier_km import as_quorum_member_with_km
from engine.rollout_orchestrator import run_canary_orchestration
from engine.policy_drilldown import load_rollout_history, summarize

def _policy(nm=1.10):
    base = {
        "trust_domains": {"example.com":5},
        "trusted_domains": ["example.com"],
        "signing_keys": {"root":{"secret_hex":"aa"*32,"algo":"sha256"}},
        "min_distinct_sources": 1,
        "min_total_trust": 1,
        "min_provenance_level": 2,
        "freshness_sla_sec_by_type": {"kpi": 600, "latency": 600},
        "default_freshness_sec": 1200,
        "perf_sla": {
            "latency_ms": {"p95_max": 150.0, "p99_max": 300.0},
            "throughput_rps": {"min": 100.0},
            "error_rate": {"max": 0.02},
            "near_miss_factor": nm
        },
        "canary_autotune": {
            "accel_threshold": 1.4,
            "decel_threshold": 1.0,
            "accel_factor": 2.0,
            "decel_factor": 0.5,
            "min_step": 1,
            "max_step": 100
        }
    }
    return strict_prod_from(json.dumps(base))

def _http_ok(url: str, method: str):
    return (200, {"date":"Tue, 01 Jul 2025 12:00:00 GMT"}, b"{}")

def _mk_bundle(pol, kid, secret_hex, claims):
    pol2 = json.loads(json.dumps(pol))
    pol2["signing_keys"] = {kid: {"secret_hex": secret_hex, "algo":"sha256"}}
    prod = ensure_proof_and_package(response_text="OK", claims=claims, policy=pol2, http_fetcher=_http_ok, sign_key_id=kid)
    assert prod["ok"]
    return prod["proof"]

def test_nearmiss_conservative_autotune(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    pol = _policy(nm=1.20)  # דורש 20% מרווח → near-miss מורגש יותר
    kr = keyring_from_policy(pol)
    exp = time.time() + 3600
    stmt = issue_delegation("root", pol["signing_keys"]["root"]["secret_hex"], child_kid="canary", scopes=["deploy"], exp_epoch=exp)
    child = derive_child_secret_hex(pol["signing_keys"]["root"]["secret_hex"], "canary", salt_hex=stmt["salt_hex"])
    km = KeychainManager(kr, lambda: [stmt])
    v = as_quorum_member_with_km(km, expected_scope="deploy")

    now = time.time()
    # Headroom נוח אך לא ענק: p95=140 (סף 150) → headroom≈1.07 < 1.20 (near-miss)
    base_claims = [{
        "id":"lat:p95","type":"latency","quantile":"p95","text":"p95=140ms",
        "schema":{"type":"number","unit":"ms"}, "value":140.0,
        "evidence":[{"kind":"http","url":"https://example.com/lat","ts": now - 60}]
    },{
        "id":"thr:rps","type":"kpi","text":"rps=150",
        "schema":{"type":"number","unit":"rps"}, "value":150.0,
        "evidence":[{"kind":"http","url":"https://example.com/rps","ts": now - 60}]
    },{
        "id":"err:rate","type":"error_rate","text":"error_rate=0.01",
        "schema":{"type":"number"}, "value":0.01,
        "evidence":[{"kind":"http","url":"https://example.com/err","ts": now - 60}]
    }]
    bundle = _mk_bundle(pol, "canary", child, base_claims)

    def stage_claims(name: str, percent: int) -> List[Dict[str,Any]]:
        # נקבע headroom קבוע ~1.07 כך שתמיד near-miss
        return base_claims

    stages = [{"name":"1%","percent":1,"min_hold_sec":0},
              {"name":"5%","percent":5,"min_hold_sec":0},
              {"name":"10%","percent":10,"min_hold_sec":0}]

    out = run_canary_orchestration(
        bundle=bundle, policy=pol, verifiers=[v], expected_scope="deploy",
        k=1, stages=stages, get_stage_claims=stage_claims, autotune=True
    )
    assert out["ok"]
    # בדיקת audit: מצופה "mode":"near_miss_conservative" ולא "adaptive"
    hist = load_rollout_history()
    modes = [ev.get("mode") for ev in hist if ev.get("evt")=="autotune"]
    assert "near_miss_conservative" in modes
    # Drilldown מסכם headroom נמוך אך עקבי
    summary = summarize(hist)
    assert summary["worst_stage"] is not None