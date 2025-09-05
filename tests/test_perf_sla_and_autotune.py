# imu_repo/tests/test_perf_sla_and_autotune.py
from __future__ import annotations
import os, json, time
from typing import List, Dict, Any
from engine.policy_compiler import strict_prod_from, keyring_from_policy
from engine.respond_guard import ensure_proof_and_package
from engine.key_delegation import issue_delegation, derive_child_secret_hex
from engine.keychain_manager import KeychainManager
from engine.verifier_km import as_quorum_member_with_km
from engine.rollout_orchestrator import run_canary_orchestration

def _policy():
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
            "error_rate": {"max": 0.02}
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

def test_autotune_accelerates_and_sla_blocks(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    pol = _policy()
    kr = keyring_from_policy(pol)
    exp = time.time() + 3600
    stmt = issue_delegation("root", pol["signing_keys"]["root"]["secret_hex"], child_kid="canary", scopes=["deploy"], exp_epoch=exp)
    child = derive_child_secret_hex(pol["signing_keys"]["root"]["secret_hex"], "canary", salt_hex=stmt["salt_hex"])
    km = KeychainManager(kr, lambda: [stmt])

    # Bundle עם ראיות חתומות "לגיטימיות" — תוכן התגובה לא חשוב כאן
    now = time.time()
    base_claims = [{
        "id":"lat:p95","type":"latency","quantile":"p95","text":"p95=80ms",
        "schema":{"type":"number","unit":"ms","min":0,"max":10000},"value":80.0,
        "evidence":[{"kind":"http","url":"https://example.com/lat","ts": now - 60}],
        "consistency_group":"lat"
    },{
        "id":"thr:rps","type":"kpi","text":"rps=220",
        "schema":{"type":"number","unit":"rps","min":0,"max":100000},"value":220.0,
        "evidence":[{"kind":"http","url":"https://example.com/rps","ts": now - 60}],
        "consistency_group":"thr"
    },{
        "id":"err:rate","type":"error_rate","text":"error_rate=0.005",
        "schema":{"type":"number","unit":"","min":0,"max":1},"value":0.005,
        "evidence":[{"kind":"http","url":"https://example.com/err","ts": now - 60}],
        "consistency_group":"err"
    }]
    bundle = _mk_bundle(pol, "canary", child, base_claims)

    v = as_quorum_member_with_km(km, expected_scope="deploy")

    # פונקציית claims לכל שלב — מדמה headroom גבוה בתחילה, ואחר כך הרעה שמחצה SLA
    call = {"i": 0}
    def stage_claims(name: str, percent: int) -> List[Dict[str,Any]]:
        call["i"] += 1
        if call["i"] <= 2:
            # מעולה: p95=80 (סף 150) → headroom ~1.875; throughput=220 (סף 100) → headroom >= 2.2
            return base_claims
        else:
            # הידרדרות: p95=190 → חורג מן הסף 150 => ייזרק PerfSlaError והשלב ייכשל
            bad = json.loads(json.dumps(base_claims))
            for c in bad:
                if c["id"]=="lat:p95":
                    c["value"] = 190.0
            return bad

    stages = [{"name":"1%","percent":1,"min_hold_sec":0},{"name":"10%","percent":10,"min_hold_sec":0},{"name":"100%","percent":100,"min_hold_sec":0}]

    out = run_canary_orchestration(
        bundle=bundle, policy=pol, verifiers=[v], expected_scope="deploy",
        k=1, stages=stages, get_stage_claims=stage_claims, autotune=True
    )
    assert out["ok"]
    # בהתחלה אמור להאיץ (headroom גבוה), אחר כך ייכשל על SLA ויעצור/יחזור אחורה
    hist = out["history"]
    assert any(h.get("gate")=="pass" and "perf_headroom" in h for h in hist)
    assert any(h.get("gate")=="fail" for h in hist)