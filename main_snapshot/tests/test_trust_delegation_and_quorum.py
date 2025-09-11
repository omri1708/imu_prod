# imu_repo/tests/test_trust_delegation_and_quorum.py
from __future__ import annotations
import os, json, time
from engine.policy_compiler import strict_prod_from, keyring_from_policy
from engine.respond_guard import ensure_proof_and_package
from engine.verifier import verify_bundle_with_chain, as_quorum_member, as_quorum_member_with_chain
from engine.key_delegation import issue_delegation, derive_child_secret_hex

def _http_ok(url: str, method: str):
    return (200, {"date":"Tue, 01 Jul 2025 12:00:00 GMT"}, b"{}")

def test_delegation_chain_verification(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    base = {
        "trust_domains": {"example.com":5},
        "trusted_domains": ["example.com"],
        "signing_keys": {"root":{"secret_hex":"aa"*32,"algo":"sha256"}},  # ב־producer נשתמש בילד
        "min_distinct_sources": 1,
        "min_total_trust": 1,
        "min_provenance_level": 2
    }
    pol = strict_prod_from(json.dumps(base))
    root_keyring = keyring_from_policy(pol)  # verifier מחזיק רק root

    # ננפיק האצלה root -> teamA
    exp = time.time() + 3600
    stmt = issue_delegation("root", base["signing_keys"]["root"]["secret_hex"], child_kid="teamA", scopes=["respond","rollup"], exp_epoch=exp)
    # בצד ה-producer: נסיק סוד ילד לגיטימי
    child_secret = derive_child_secret_hex(base["signing_keys"]["root"]["secret_hex"], "teamA", salt_hex=stmt["salt_hex"])
    # המדיניות אצל ה-producer תחזיק child key (כדי שיוכל לחתום בפועל)
    prod_pol = json.loads(json.dumps(pol))
    prod_pol["signing_keys"] = {"teamA":{"secret_hex": child_secret, "algo":"sha256"}}

    claims = [{
        "id":"lat:p95",
        "type":"latency",
        "text":"p95=90ms",
        "schema":{"type":"number","unit":"ms","min":0,"max":1000},
        "value": 90,
        "evidence":[{"kind":"http","url":"https://api.example.com/metrics"}],
        "consistency_group":"lat"
    }]

    produced = ensure_proof_and_package(response_text="OK", claims=claims, policy=prod_pol, http_fetcher=_http_ok, sign_key_id="teamA")
    assert produced["ok"]
    bundle = produced["proof"]

    # verifier: root בלבד + trust_chain
    out = verify_bundle_with_chain(bundle, pol, root_keyring=root_keyring, trust_chain=[stmt], http_fetcher=_http_ok)
    assert out["ok"]

def test_quorum_verify(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    base = {
        "trust_domains": {"example.com":5},
        "trusted_domains": ["example.com"],
        "signing_keys": {"root":{"secret_hex":"aa"*32,"algo":"sha256"}},
        "min_distinct_sources": 1,
        "min_total_trust": 1,
        "min_provenance_level": 2
    }
    import copy, json as _json
    pol = strict_prod_from(_json.dumps(base))
    root_keyring = keyring_from_policy(pol)
    stmt = issue_delegation("root", base["signing_keys"]["root"]["secret_hex"], child_kid="teamB", scopes=["respond"], exp_epoch=time.time()+3600)
    child_secret = derive_child_secret_hex(base["signing_keys"]["root"]["secret_hex"], "teamB", salt_hex=stmt["salt_hex"])

    prod_pol = copy.deepcopy(pol)
    prod_pol["signing_keys"] = {"teamB":{"secret_hex": child_secret, "algo":"sha256"}}

    claims = [{
        "id":"m:kpi",
        "type":"kpi",
        "text":"throughput=100",
        "schema":{"type":"number","unit":"rps","min":0,"max":10000},
        "value": 100,
        "evidence":[{"kind":"http","url":"https://example.com/kpi"}],
        "consistency_group":"kpi"
    }]

    produced = ensure_proof_and_package(response_text="OK", claims=claims, policy=prod_pol, http_fetcher=_http_ok, sign_key_id="teamB")
    bundle = produced["proof"]

    # נגדיר 3 מאמתים: שניים עם שרשרת תקפה, אחד "קשוח מדי"
    v1 = as_quorum_member_with_chain(root_keyring, [stmt], http_fetcher=_http_ok)
    v2 = as_quorum_member_with_chain(root_keyring, [stmt], http_fetcher=_http_ok)

    # מאמת שלישי עם מדיניות שמחייבת min_distinct_sources=2 → ייכשל
    pol_harsh = copy.deepcopy(pol)
    pol_harsh["min_distinct_sources"] = 2
    def v3(bundle, policy):
        from engine.verifier import verify_bundle_with_chain
        try:
            out = verify_bundle_with_chain(bundle, pol_harsh, root_keyring=root_keyring, trust_chain=[stmt], http_fetcher=_http_ok)
            return {"ok": True, **out}
        except Exception as e:
            return {"ok": False, "reason": str(e)}

    from engine.quorum_verify import quorum_verify
    out = quorum_verify(bundle, pol, verifiers=[v1, v2, v3], k=2)
    assert out["ok"] and out["oks"] >= 2