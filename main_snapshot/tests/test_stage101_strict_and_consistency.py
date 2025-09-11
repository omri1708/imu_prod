# imu_repo/tests/test_stage101_strict_and_consistency.py
from __future__ import annotations
import os, json, time
from typing import List, Dict, Any
from engine.policy_compiler import strict_prod_from, keyring_from_policy
from engine.strict_mode import strict_package_response, StrictGroundingError
from engine.key_delegation import issue_delegation, derive_child_secret_hex
from engine.keychain_manager import KeychainManager
from engine.verifier_km import as_quorum_member_with_km
from engine.rollout_orchestrator import run_canary_orchestration
from engine.policy_drilldown import load_rollout_history, summarize

def _policy():
    base = {
        "trust_domains": {"example.com":5},
        "trusted_domains": ["example.com"],
        "signing_keys": {"root":{"secret_hex":"aa"*32,"algo":"sha256"}},
        "min_distinct_sources": 1,
        "min_total_trust": 1,
        "min_provenance_level": 1,
        "default_freshness_sec": 1200,
        "perf_sla": {
            "latency_ms": {"p95_max": 150.0},
            "throughput_rps": {"min": 100.0},
            "error_rate": {"max": 0.05},
            "near_miss_factor": 1.15
        },
        "canary_autotune": {
            "accel_threshold": 1.4,
            "decel_threshold": 1.0,
            "accel_factor": 2.0,
            "decel_factor": 0.5,
            "min_step": 1,
            "max_step": 100
        },
        "consistency": {
            "drift_pct": 0.10,  # 10%
            "near_miss_streak_heal_threshold": 2,
            "heal_action": "freeze_autotune"
        }
    }
    return strict_prod_from(json.dumps(base))

def _http_ok(url: str, method: str):
    return (200, {"date":"Tue, 01 Jul 2025 12:00:00 GMT"}, b"{}")

def test_strict_mode_creates_compute_claim(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    os.environ["IMU_STATE_DIR"] = str(tmp_path / ".state")
    pol = _policy()

    # אריזה עם claims=[]
    proof = strict_package_response(
        response_text="42",
        claims=[],
        policy=pol,
        http_fetcher=_http_ok,
        sign_key_id="root"  # במימוש compiler המפתח root זמין במדיניות
    )
    assert isinstance(proof, dict)
    # צריך להכיל claims שנוצרו (compute)
    cl = proof.get("claims") or proof.get("metrics") or []
    assert isinstance(cl, list) and any(c.get("type")=="compute" for c in cl)

def test_consistency_nearmiss_freezes_autotune(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    os.environ["IMU_STATE_DIR"] = str(tmp_path / ".state")
    pol = _policy()
    kr = keyring_from_policy(pol)
    exp = time.time() + 3600
    from engine.respond_guard import ensure_proof_and_package
    from engine.verifier_km import as_quorum_member_with_km
    from engine.key_delegation import issue_delegation, derive_child_secret_hex

    stmt = issue_delegation("root", pol["signing_keys"]["root"]["secret_hex"], child_kid="canary", scopes=["deploy"], exp_epoch=exp)
    child = derive_child_secret_hex(pol["signing_keys"]["root"]["secret_hex"], "canary", salt_hex=stmt["salt_hex"])
    km = KeychainManager(kr, lambda: [stmt])
    v = as_quorum_member_with_km(km, expected_scope="deploy")

    now = time.time()
    base_claims = [{
        "id":"lat","type":"latency","quantile":"p95","text":"p95=140ms",
        "schema":{"type":"number","unit":"ms"},"value":140.0,
        "evidence":[{"kind":"http","url":"https://example.com/lat","ts": now - 60}],
        "consistency_group":"lat"
    },{
        "id":"thr","type":"kpi","text":"rps=120","schema":{"type":"number","unit":"rps"},"value":120.0,
        "evidence":[{"kind":"http","url":"https://example.com/rps","ts": now - 60}],
        "consistency_group":"thr"
    }]

    # חבילה חתומה בסיסית
    prod = ensure_proof_and_package(response_text="OK", claims=base_claims, policy=pol, http_fetcher=_http_ok, sign_key_id="root")
    bundle = prod["proof"]

    # יצירת מצב near-miss רציף: headroom ~ 150/140=1.07 < 1.15
    calls = {"i":0}
    def stage_claims(name: str, percent: int) -> List[Dict[str,Any]]:
        calls["i"] += 1
        # נייצר גם drift קטן בסיבוב השלישי (lat מתעדכן ל-160 → drift≈14% > 10%)
        cc = json.loads(json.dumps(base_claims))
        if calls["i"] >= 3 and name != "1%":
            for c in cc:
                if c["id"]=="lat":
                    c["value"] = 160.0
        return cc

    stages = [{"name":"1%","percent":1,"min_hold_sec":0},{"name":"5%","percent":5,"min_hold_sec":0},{"name":"10%","percent":10,"min_hold_sec":0}]
    out = run_canary_orchestration(
        bundle=bundle, policy=pol, verifiers=[v], expected_scope="deploy",
        k=1, stages=stages, get_stage_claims=stage_claims, autotune=True
    )
    assert out["ok"]
    # ציפה: heal freeze_autotune נרשם ב-audit
    from engine.policy_drilldown import load_rollout_history
    hist = load_rollout_history()
    assert any(ev.get("evt")=="heal" and ev.get("mode")=="freeze_autotune" for ev in hist)