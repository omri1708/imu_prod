# imu_repo/tests/test_stage76_ab_bestofall.py
from __future__ import annotations
import os, glob, asyncio

from engine.config import load_config, save_config
from user_model.policy import set_profile
from grounded.provenance import STORE
from engine.synthesis_pipeline import run_pipeline
from engine.phi import max_allowed

def _reset():
    os.makedirs(STORE, exist_ok=True)
    for p in glob.glob(os.path.join(STORE, "*.json")):
        os.remove(p)

async def _run_happy():
    _reset()
    cfg = load_config()
    cfg.setdefault("evidence", {}).update({"required": True})
    cfg.setdefault("guard", {}).update({"min_trust": 0.7, "max_age_s": 3600})
    # הקל מעט את max_allowed כדי לאפשר מעבר רגיל
    cfg.setdefault("phi", {}).update({"max_allowed": 50_000.0})
    save_config(cfg)

    set_profile("u_ab", min_trust=0.7, max_age_s=3600, strict_grounded=True)

    spec = {"name":"ab_app","goal":"Hello A/B!"}
    out = await run_pipeline(spec, user_id="u_ab")

    assert isinstance(out, dict) and "text" in out and "claims" in out
    assert "VARIANT=A" in out["text"], "המהירה (A) אמורה לנצח"
    files = glob.glob(os.path.join(STORE, "*.json"))
    assert files, "צריך היווצרו CAS evidences חתומים"

async def _run_rollback():
    _reset()
    cfg = load_config()
    cfg.setdefault("evidence", {}).update({"required": True})
    cfg.setdefault("guard", {}).update({"min_trust": 0.7, "max_age_s": 3600})
    # אולטרה-מחמיר: סף נמוך שיגרום לרולבאק
    cfg.setdefault("phi", {}).update({"max_allowed": 0.1})
    save_config(cfg)

    set_profile("u_ab2", min_trust=0.7, max_age_s=3600, strict_grounded=True)

    spec = {"name":"ab_app2","goal":"Hello A/B!"}
    failed = False
    try:
        await run_pipeline(spec, user_id="u_ab2")
    except RuntimeError as e:
        # מצופה "ab_worse_than_allowed"
        failed = "ab_worse_than_allowed" in str(e)
    assert failed, "עם סף Φ קיצוני צריך רולבאק/כשל"

def run():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(_run_happy())
    loop.run_until_complete(_run_rollback())
    print("OK")
    return 0

if __name__=="__main__":
    raise SystemExit(run())