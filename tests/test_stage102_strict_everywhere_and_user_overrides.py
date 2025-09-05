# imu_repo/tests/test_stage102_strict_everywhere_and_user_overrides.py
from __future__ import annotations
import time, json, os
from typing import Dict, Any, List, Tuple
from engine.synthesis_pipeline import SynthesisPipeline
from engine.policy_compiler import strict_prod_from, keyring_from_policy
from engine.verifier_km import as_quorum_member_with_policy
from engine.policy_overrides import apply_user_overrides

def _base_policy() -> Dict[str,Any]:
    base = {
        "trust_domains": {"example.com": 5},
        "trusted_domains": ["example.com"],
        "signing_keys": {"root": {"secret_hex": "aa"*32, "algo":"sha256"}},
        "min_distinct_sources": 1,
        "min_total_trust": 1,
        "min_provenance_level": 1,
        "default_freshness_sec": 600,
        "perf_sla": {
            "latency_ms": {"p95_max": 150.0},
            "throughput_rps": {"min": 100.0},
            "error_rate": {"max": 0.05},
            "near_miss_factor": 1.15
        },
        "consistency": {
            "drift_pct": 0.10,
            "near_miss_streak_heal_threshold": 3,
            "heal_action": "freeze_autotune"
        }
    }
    return strict_prod_from(json.dumps(base))

def _http_ok(url: str, method: str):
    return (200, {"date":"Tue, 01 Jul 2025 12:00:00 GMT"}, b"{}")

def _gen_no_claims(ctx: Dict[str,Any]) -> Tuple[str, None]:
    # מחזיר טקסט בלבד — המעטפת תיצור compute-claim דטרמיניסטי
    return ("the answer is 42", None)

def _gen_with_claims(ctx: Dict[str,Any]) -> Tuple[str, List[Dict[str,Any]]]:
    now = time.time()
    return ("latency ok", [{
        "id":"lat","type":"latency","quantile":"p95","text":"p95=120ms",
        "schema":{"type":"number","unit":"ms"},"value":120.0,
        "evidence":[{"kind":"http","url":"https://example.com/lat","ts": now - 60}],
        "consistency_group":"lat"
    }])

def test_user_overrides_apply_and_strict_packing(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    os.environ["IMU_STATE_DIR"] = str(tmp_path / ".state")

    base = _base_policy()
    pipe = SynthesisPipeline(base_policy=base, http_fetcher=_http_ok, sign_key_id="root")

    # פרופיל רגולטורי קשיח → ספי p95/Trust מוחמרים
    ctx = {"user":{"tier":"enterprise","risk_score":0.7}}
    v = as_quorum_member_with_policy(base, expected_scope="deploy")

    # 1) מחייב compute-claim כשאין claims
    run1 = pipe.run_once(
        ctx=ctx, generate_fn=_gen_no_claims, verifiers=[v],
        rollout_stages=[{"name":"1%","percent":1},{"name":"10%","percent":10}],
        expected_scope="deploy", k=1, autotune=False
    )
    assert run1["ok"] and isinstance(run1["bundle"], dict)

    # 2) כשיש claims — עובר אימות ו־rollout בסיסי
    run2 = pipe.run_once(
        ctx=ctx, generate_fn=_gen_with_claims, verifiers=[v],
        rollout_stages=[{"name":"1%","percent":1},{"name":"10%","percent":10}],
        expected_scope="deploy", k=1, autotune=True
    )
    assert run2["ok"]
    pol_eff = run2["policy"]
    # ודא שה־overrides הוחלו (p95 מקשיח מ-150 למשהו ≤100)
    assert float(pol_eff["perf_sla"]["latency_ms"]["p95_max"]) <= 100.0

def test_standard_user_defaults(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    os.environ["IMU_STATE_DIR"] = str(tmp_path / ".state")
    base = _base_policy()
    ctx = {"user":{"tier":"standard"}}
    eff = apply_user_overrides(base, ctx["user"])
    # ברירת־מחדל p95 נשארת סביב 150
    assert abs(float(eff["perf_sla"]["latency_ms"]["p95_max"]) - 150.0) < 1e-6