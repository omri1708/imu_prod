# imu_repo/tests/test_canary_stages_and_freshness.py
from __future__ import annotations
import os, json, time, copy
from engine.policy_compiler import strict_prod_from, keyring_from_policy
from engine.respond_guard import ensure_proof_and_package
from engine.key_delegation import issue_delegation, derive_child_secret_hex
from engine.verifier_km import as_quorum_member_with_km
from engine.keychain_manager import KeychainManager
from engine.rollout_orchestrator import run_canary_orchestration

def _http_ok(url: str, method: str):
    return (200, {"date":"Tue, 01 Jul 2025 12:00:00 GMT"}, b"{}")

def _policy_with_freshness():
    base = {
        "trust_domains": {"example.com":5},
        "trusted_domains": ["example.com"],
        "signing_keys": {"root":{"secret_hex":"aa"*32,"algo":"sha256"}},
        "min_distinct_sources": 1,
        "min_total_trust": 1,
        "min_provenance_level": 2,
        "freshness_sla_sec_by_type": {"kpi": 600, "latency": 600},
        "default_freshness_sec": 1200
    }
    return strict_prod_from(json.dumps(base))

def _make_bundle(pol, kid: str, secret_hex: str, claims):
    prod_pol = json.loads(json.dumps(pol))
    prod_pol["signing_keys"] = {kid: {"secret_hex": secret_hex, "algo":"sha256"}}
    produced = ensure_proof_and_package(response_text="OK", claims=claims, policy=prod_pol, http_fetcher=_http_ok, sign_key_id=kid)
    assert produced["ok"]
    return produced["proof"]

def test_freshness_enforced_ok_and_stale(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    pol = _policy_with_freshness()
    root_keyring = keyring_from_policy(pol)

    # Delegation for deploy scope
    exp = time.time() + 3600
    stmt = issue_delegation("root", pol["signing_keys"]["root"]["secret_hex"], child_kid="deployer", scopes=["deploy"], exp_epoch=exp)
    child_secret = derive_child_secret_hex(pol["signing_keys"]["root"]["secret_hex"], "deployer", salt_hex=stmt["salt_hex"])

    now = time.time()
    fresh_claims = [{
        "id":"kpi:tps","type":"kpi","text":"tps=150",
        "schema":{"type":"number","unit":"rps","min":0,"max":10000},"value":150,
        "evidence":[{"kind":"http","url":"https://example.com/kpi","ts": now - 120}],  # טרי (מתחת 600)
        "consistency_group":"kpi"
    }]
    stale_claims = [{
        "id":"kpi:tps","type":"kpi","text":"tps=150",
        "schema":{"type":"number","unit":"rps","min":0,"max":10000},"value":150,
        "evidence":[{"kind":"http","url":"https://example.com/kpi","ts": now - 99999}],  # ישן
        "consistency_group":"kpi"
    }]

    fresh_bundle = _make_bundle(pol, "deployer", child_secret, fresh_claims)
    stale_bundle = _make_bundle(pol, "deployer", child_secret, stale_claims)

    # Keychain manager שיחזיר את השרשרת
    chain_store = [stmt]
    def provider(): return list(chain_store)
    km = KeychainManager(root_keyring, provider)

    v1 = as_quorum_member_with_km(km, http_fetcher=_http_ok, expected_scope="deploy")
    v2 = as_quorum_member_with_km(km, http_fetcher=_http_ok, expected_scope="deploy")

    # Canary plan זריז לבדיקות
    stages = [{"name":"5%","percent":5,"min_hold_sec":0}, {"name":"50%","percent":50,"min_hold_sec":0}, {"name":"100%","percent":100,"min_hold_sec":0}]

    # טרי — עובר כל השלבים
    out_ok = run_canary_orchestration(bundle=fresh_bundle, policy=pol, verifiers=[v1,v2], expected_scope="deploy", k=2, stages=stages)
    assert out_ok["ok"] and out_ok["completed"] and not out_ok["aborted"]
    assert out_ok["final_stage"] == "100%"

    # ישן — יכשל בשער הראשון ויבצע רולבאק/עצירה
    try:
        out_bad = run_canary_orchestration(bundle=stale_bundle, policy=pol, verifiers=[v1,v2], expected_scope="deploy", k=2, stages=stages)
        # הפונקציה לא זורקת—היא מתעדת כישלון, ולא מתקדמת
        assert out_bad["ok"] and (out_bad["aborted"] or out_bad["history"] and out_bad["history"][0]["gate"]=="fail")
    except Exception as e:
        # אם בחרת להחריף gate_release לזרוק—זה גם תקף בבדיקה: העיקר שטריות נאכפת
        assert "freshness" in str(e).lower() or "stale" in str(e).lower()

def test_canary_backoff_progression(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    pol = _policy_with_freshness()
    root_keyring = keyring_from_policy(pol)
    exp = time.time() + 3600
    stmt = issue_delegation("root", pol["signing_keys"]["root"]["secret_hex"], child_kid="shipper2", scopes=["deploy"], exp_epoch=exp)
    child_secret = derive_child_secret_hex(pol["signing_keys"]["root"]["secret_hex"], "shipper2", salt_hex=stmt["salt_hex"])

    now = time.time()
    claims = [{
        "id":"lat:p95","type":"latency","text":"p95=88ms",
        "schema":{"type":"number","unit":"ms","min":0,"max":1000},"value":88,
        "evidence":[{"kind":"http","url":"https://example.com/lat","ts": now - 60}],
        "consistency_group":"lat"
    }]
    bundle = _make_bundle(pol, "shipper2", child_secret, claims)

    chain_store = [stmt]
    def provider(): return list(chain_store)
    km = KeychainManager(root_keyring, provider)

    # נבנה שני מאמתים: אחד "יעיל" ואחד שמדי פעם נכשל כדי לדמות backoff
    pass_every_call = as_quorum_member_with_km(km, expected_scope="deploy")
    fail_toggle = {"i":0}
    def flaky(bundle_, policy_):
        fail_toggle["i"] += 1
        if fail_toggle["i"] % 2 == 0:
            return {"ok": False, "reason":"flaky"}
        return {"ok": True}

    stages = [{"name":"1%","percent":1,"min_hold_sec":0},{"name":"10%","percent":10,"min_hold_sec":0},{"name":"100%","percent":100,"min_hold_sec":0}]

    out = run_canary_orchestration(bundle=bundle, policy=pol, verifiers=[pass_every_call, flaky], expected_scope="deploy", k=1, stages=stages)
    # בגלל ה-flaky, ייתכן שלא נגיע ל-100% מיד, אבל לא אמורים להיכשל סופית.
    assert out["ok"] and not out["aborted"]