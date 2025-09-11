# imu_repo/tests/test_stage85_official_verifiers.py
from __future__ import annotations
import os, glob, asyncio

from grounded.provenance import STORE
from grounded.claims import current
from engine.config import load_config, save_config
from engine.synthesis_pipeline import run_pipeline
from engine.learn_store import LEARN_DIR
from user_model.policy import set_profile
from verifiers.official_registry import register_official_source, sign_for_source

def _reset_all():
    os.makedirs(STORE, exist_ok=True)
    for p in glob.glob(os.path.join(STORE, "*.json")):
        os.remove(p)
    os.makedirs(LEARN_DIR, exist_ok=True)
    for p in glob.glob(os.path.join(LEARN_DIR, "*")):
        os.remove(p)
    current().reset()

def _set_cfg(require_official: bool):
    cfg = load_config()
    cfg["evidence"] = {"required": True, "signing_secret": "stage85_secret"}
    # guard כללי: דורש spec+plan; threshold אמון סביר
    cfg["guard"] = {"min_trust": 0.7, "max_age_s": 3600.0, "min_count": 2, "required_kinds": ["spec","plan"]}
    # official: אם נדרוש — נוסיף גם kind של official_verified לרשימת החובה
    if require_official:
        rk = list(cfg["guard"]["required_kinds"])
        if "official_verified" not in rk:
            rk.append("official_verified")
        cfg["guard"]["required_kinds"] = rk
    cfg["phi"] = {"max_allowed": 50_000.0}
    save_config(cfg)

async def test_official_passes_with_valid_signature():
    _reset_all()
    _set_cfg(require_official=True)

    # רושמים מקור רשמי gov_il עם סוד
    register_official_source("gov_il", shared_secret="topsecret_gov", trust=0.99)

    # מזריקים ראיה "רשמית" עם חתימה נכונה
    data = {"user_id": 123, "status": "eligible", "version": 1}
    sig = sign_for_source("gov_il", data)
    current().add_evidence("gov_record", {
        "source_url": "official://gov_il/api",
        "trust": 0.95,
        "ttl_s": 1200,
        "payload": {"data": data, "official": {"source_id": "gov_il", "signature": sig}}
    })

    spec = {"name":"benefits_service", "goal":"determine eligibility realtime"}
    set_profile("u", strict_grounded=True,
                phi_weights={"errors":0.25,"distrust":0.2,"latency":0.4,"cost":0.1,"energy":0.03,"memory":0.02})

    out = await run_pipeline(spec, user_id="u", learn=True)
    # לא נפלנו לפולבאק — כי official_verified נוסף ועבר Guard
    assert "text" in out and "[FALLBACK]" not in out["text"]
    kinds = {e.get("kind") for e in current().snapshot()}
    assert "official_verified" in kinds

async def test_official_blocks_on_bad_signature():
    _reset_all()
    _set_cfg(require_official=True)
    register_official_source("gov_il", shared_secret="topsecret_gov", trust=0.99)

    # חתימה שגויה
    data = {"user_id": 999, "status": "rejected", "version": 3}
    bad_sig = "not_the_right_sig"
    current().add_evidence("gov_record", {
        "source_url": "official://gov_il/api",
        "trust": 0.95,
        "ttl_s": 1200,
        "payload": {"data": data, "official": {"source_id": "gov_il", "signature": bad_sig}}
    })

    spec = {"name":"benefits_service", "goal":"determine eligibility realtime"}
    set_profile("u2", strict_grounded=True,
                phi_weights={"errors":0.25,"distrust":0.2,"latency":0.4,"cost":0.1,"energy":0.03,"memory":0.02})

    out = await run_pipeline(spec, user_id="u2", learn=True)
    # נחסם — כי official_verified לא התקבל (נוצר official_verification_failed) וה-Guard דורש אותו
    assert "text" in out and "[FALLBACK]" in out["text"]
    kinds = {e.get("kind") for e in current().snapshot()}
    assert "official_verification_failed" in kinds
    assert "official_verified" not in kinds

def run():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_official_passes_with_valid_signature())
    loop.run_until_complete(test_official_blocks_on_bad_signature())
    print("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())