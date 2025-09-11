from __future__ import annotations
import os, glob, asyncio

from grounded.provenance import STORE
from grounded.claims import current
from grounded.provenance_confidence import register_source, set_source_trust, effective_session_trust
from engine.config import load_config, save_config
from engine.synthesis_pipeline import run_pipeline
from engine.learn_store import LEARN_DIR
from user_model.policy import set_profile

def _reset_all():
    os.makedirs(STORE, exist_ok=True)
    for p in glob.glob(os.path.join(STORE, "*.json")):
        os.remove(p)
    os.makedirs(LEARN_DIR, exist_ok=True)
    for p in glob.glob(os.path.join(LEARN_DIR, "*")):
        os.remove(p)
    # איפוס קונטקסט
    current().reset()
    # אפס קונפיג
    cfg = load_config()
    cfg["evidence"] = {"required": True, "signing_secret": "stage83_secret"}
    cfg["guard"] = {"min_trust": 0.7, "max_age_s": 3600}
    cfg["phi"] = {"max_allowed": 50_000.0}
    save_config(cfg)

async def test_signed_evidence_and_session_trust_weighting():
    _reset_all()
    # רישום מקורות: אחד חזק, אחד חלש
    register_source("source_local", trust=0.95, prefixes=["local://"])  # מחליף/מבסס דיפולט
    register_source("source_weak", trust=0.3, prefixes=["weak://"])
    set_source_trust("source_weak", 0.25)

    # מוסיפים ידנית ראיות מסוגים שונים
    current().add_evidence("spec", {"source_url":"local://spec", "trust":0.95, "ttl_s":600, "payload":{"ok":True}})
    current().add_evidence("hint", {"source_url":"weak://rumor", "trust":0.50, "ttl_s":600, "payload":{"note":"unverified"}})

    # מחשבים אמון אפקטיבי – אמור להיות בין 0.6 ל-0.9, קרוב ל-local בזכות משקל־טריות+דומיננטיות
    st = effective_session_trust(current().snapshot())
    assert 0.6 <= st <= 0.95

async def test_ab_selector_penalizes_low_session_trust():
    _reset_all()
    register_source("source_local", trust=0.95, prefixes=["local://"])
    register_source("source_weak", trust=0.1, prefixes=["weak://"])

    # מזריקים ראיה חלשה כדי להוריד אמון־סשן
    current().add_evidence("weak_hint", {"source_url":"weak://gossip", "trust":0.2, "ttl_s":600, "payload":{"h":"g"}})
    # spec ו-plan/… יתווספו במהלך הריצה

    spec = {"name":"realtime_stream", "goal":"Serve users < 40ms realtime"}
    # העדפות שמדגישות אמינות ומהירות
    set_profile("u", strict_grounded=True, phi_weights={"errors":0.3,"distrust":0.2,"latency":0.4,"cost":0.08,"energy":0.01,"memory":0.01})

    # ריצה—גם אם תופיע וריאצית Explore, ענישת אמון־סשן תטה לבחירת A
    out = await run_pipeline(spec, user_id="u", learn=True)
    assert "VARIANT=A" in out["text"]

def run():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_signed_evidence_and_session_trust_weighting())
    loop.run_until_complete(test_ab_selector_penalizes_low_session_trust())
    print("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())