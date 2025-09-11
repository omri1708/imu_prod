# imu_repo/tests/test_stage86_multi_tenant_privacy.py
from __future__ import annotations
import os, json, time, glob, asyncio

from grounded.provenance import STORE
from grounded.claims import current
from engine.user_scope import user_scope
from engine.config import load_config, save_config
from engine.synthesis_pipeline import run_pipeline
from user_model.consent import set_consent, get_consent
from user_model.profile_store import set_pref, get_profile, consolidate
from privacy.storage import save_json_encrypted, load_json_encrypted, purge_expired

def _reset_prov():
    os.makedirs(STORE, exist_ok=True)
    for p in glob.glob(os.path.join(STORE, "*.json")):
        os.remove(p)

async def test_user_isolation_in_provenance():
    _reset_prov()
    with user_scope("alice"):
        current().add_evidence("spec", {"source_url":"local://spec","trust":0.95,"ttl_s":600,"payload":{"k":1}})
    with user_scope("bob"):
        current().add_evidence("spec", {"source_url":"local://spec","trust":0.95,"ttl_s":600,"payload":{"k":2}})

    files = sorted(glob.glob(os.path.join(STORE, "*__*.json")))
    assert any("__alice__" in f for f in files)
    assert any("__bob__" in f for f in files)

async def test_encrypted_profile_and_contradictions():
    with user_scope("alice"):
        set_consent("alice", personalization=True, cross_session_learning=True)
        set_pref("alice", "theme", "dark", confidence=0.7)
        set_pref("alice", "theme", "light", confidence=0.9)  # סתירה — החדש גובר
        prof = consolidate("alice")
        assert prof["prefs"]["theme"]["value"] == "light"
        assert len(prof.get("contradictions", [])) >= 1

async def test_encrypted_store_and_ttl():
    # נשמור אובייקט קצר מועד
    save_json_encrypted("bob", "ephemeral", {"v": 1}, ttl_s=0.01)
    assert load_json_encrypted("bob", "ephemeral") == {"v": 1}
    time.sleep(0.02)
    # לאחר פקיעה — קריאה תחזיר None וה-purge ימחק
    assert load_json_encrypted("bob", "ephemeral") is None
    deleted = purge_expired("bob")
    assert deleted >= 1

async def test_pipeline_marks_user_and_runs():
    # מבטיחים שהרצה מציינת user ויוצרת ראיות בהתאם
    cfg = load_config()
    cfg["evidence"] = {"required": True, "signing_secret": "stage86_secret"}
    cfg["guard"] = {"min_trust": 0.7, "max_age_s": 3600.0, "min_count": 1, "required_kinds": ["spec"]}
    cfg["phi"] = {"max_allowed": 50000.0}
    save_config(cfg)

    with user_scope("carol"):
        spec = {"name": "small_service", "goal": "ok"}
        out = asyncio.get_event_loop().run_until_complete(run_pipeline(spec, user_id="carol", learn=True))
        assert "text" in out
        kinds = [e.get("kind") for e in current().snapshot()]
        # מתוך הצנרת הקיימת: spec/plan/… — לפחות spec מופיע (דרישת guard)
        assert "spec" in kinds
        # והראיות מסומנות user_id=carol
        assert all((e.get("user_id") == "carol") for e in current().snapshot())

def run():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_user_isolation_in_provenance())
    loop.run_until_complete(test_encrypted_profile_and_contradictions())
    loop.run_until_complete(test_encrypted_store_and_ttl())
    loop.run_until_complete(test_pipeline_marks_user_and_runs())
    print("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())