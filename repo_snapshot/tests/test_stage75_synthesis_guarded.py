# imu_repo/tests/test_stage75_synthesis_guarded.py
from __future__ import annotations
import os, glob, asyncio

from engine.config import load_config, save_config
from user_model.policy import set_profile
from grounded.provenance import STORE
from engine.synthesis_pipeline import run_pipeline

def _reset():
    os.makedirs(STORE, exist_ok=True)
    for p in glob.glob(os.path.join(STORE, "*.json")):
        os.remove(p)

async def _run():
    _reset()
    # אוכפים Evidences חובה + Gate גלובלי
    cfg = load_config()
    cfg.setdefault("evidence", {}).update({"required": True})
    cfg.setdefault("guard", {}).update({"min_trust": 0.7, "max_age_s": 3600})
    save_config(cfg)

    # פרופיל משתמש
    set_profile("u_syn", min_trust=0.8, max_age_s=1800, strict_grounded=True)

    spec = {"name":"hello_app","goal":"Hello, IMU!"}
    out = await run_pipeline(spec, user_id="u_syn")
    # וידוא שהפלט מחוייב claims חתומים
    assert isinstance(out, dict) and "text" in out and "claims" in out
    assert out["text"].startswith("[ARTIFACT:hello_app]")
    assert isinstance(out["claims"], list) and len(out["claims"]) >= 1

    # ודא שקבצי CAS נוצרו
    files = glob.glob(os.path.join(STORE, "*.json"))
    assert files, "expected signed provenance records"

def run():
    asyncio.get_event_loop().run_until_complete(_run())
    print("OK")
    return 0

if __name__=="__main__":
    raise SystemExit(run())