# imu_repo/tests/test_scoped_delegations_and_quorum_gate.py
from __future__ import annotations
import os, json, time, copy
from engine.policy_compiler import strict_prod_from, keyring_from_policy
from engine.respond_guard import ensure_proof_and_package
from engine.key_delegation import issue_delegation, derive_child_secret_hex
from engine.verifier import as_quorum_member_with_chain
from engine.verifier_km import as_quorum_member_with_km
from engine.keychain_manager import KeychainManager
from engine.rollout_quorum_gate import gate_release

def _http_ok(url: str, method: str):
    return (200, {"date":"Tue, 01 Jul 2025 12:00:00 GMT"}, b"{}")

def _make_policy():
    base = {
        "trust_domains": {"example.com":5},
        "trusted_domains": ["example.com"],
        "signing_keys": {"root":{"secret_hex":"aa"*32,"algo":"sha256"}},
        "min_distinct_sources": 1,
        "min_total_trust": 1,
        "min_provenance_level": 2
    }
    return strict_prod_from(json.dumps(base))

def test_scope_enforced_at_verifier(tmp_path, monkeypatch):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    pol = _make_policy()
    root_keyring = keyring_from_policy(pol)

    # root -> teamC with scope "respond" only
    exp = time.time() + 600
    stmt = issue_delegation("root", pol["signing_keys"]["root"]["secret_hex"], child_kid="teamC", scopes=["respond"], exp_epoch=exp)
    child_secret = derive_child_secret_hex(pol["signing_keys"]["root"]["secret_hex"], "teamC", salt_hex=stmt["salt_hex"])
    prod_pol = copy.deepcopy(pol); prod_pol["signing_keys"] = {"teamC":{"secret_hex": child_secret, "algo":"sha256"}}

    claims = [{
        "id":"lat:p95","type":"latency","text":"p95=80ms",
        "schema":{"type":"number","unit":"ms","min":0,"max":1000},"value":80,
        "evidence":[{"kind":"http","url":"https://example.com/metrics"}],
        "consistency_group":"lat"
    }]
    produced = ensure_proof_and_package(response_text="OK", claims=claims, policy=prod_pol, http_fetcher=_http_ok, sign_key_id="teamC")
    bundle = produced["proof"]

    # verifier דורש scope=respond → עובר
    v_ok = as_quorum_member_with_chain(root_keyring, [stmt], http_fetcher=_http_ok, expected_scope="respond")
    out = v_ok(bundle, pol); assert out["ok"]

    # verifier דורש scope=rollup → נכשל
    v_bad = as_quorum_member_with_chain(root_keyring, [stmt], http_fetcher=_http_ok, expected_scope="rollup")
    out2 = v_bad(bundle, pol); assert not out2["ok"] and "scope error" in out2["reason"]

def test_keychain_manager_auto_refresh_and_gate(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    pol = _make_policy()
    root_keyring = keyring_from_policy(pol)

    # שרשרת קצרה (תפוג בקרוב) – ה-Provider יחזיר אותה מחדש כדי לוודא ריענון
    exp = time.time() + 3
    stmt = issue_delegation("root", pol["signing_keys"]["root"]["secret_hex"], child_kid="shipper", scopes=["respond","deploy"], exp_epoch=exp)
    child_secret = derive_child_secret_hex(pol["signing_keys"]["root"]["secret_hex"], "shipper", salt_hex=stmt["salt_hex"])
    prod_pol = json.loads(json.dumps(pol)); prod_pol["signing_keys"] = {"shipper":{"secret_hex": child_secret, "algo":"sha256"}}

    chain_store = [stmt]
    def provider():
        # מחזיר את השרשרת הנוכחית (אפשר לדמיין כאן Fetch מה־KV)
        return list(chain_store)

    from engine.keychain_manager import KeychainManager
    km = KeychainManager(root_keyring, provider, refresh_margin_sec=1)
    v1 = as_quorum_member_with_km(km, http_fetcher=_http_ok, expected_scope="respond")
    v2 = as_quorum_member_with_km(km, http_fetcher=_http_ok, expected_scope="respond")

    claims = [{
        "id":"kpi:tps","type":"kpi","text":"tps=120",
        "schema":{"type":"number","unit":"rps","min":0,"max":10000},"value":120,
        "evidence":[{"kind":"http","url":"https://example.com/kpi"}],
        "consistency_group":"kpi"
    }]
    produced = ensure_proof_and_package(response_text="OK", claims=claims, policy=prod_pol, http_fetcher=_http_ok, sign_key_id="shipper")
    bundle = produced["proof"]

    # gate עם k=2
    out = gate_release(bundle, pol, verifiers=[v1, v2], k=2, expected_scope="respond")
    assert out["ok"] and out["oks"] == 2