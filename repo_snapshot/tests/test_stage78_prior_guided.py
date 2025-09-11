# imu_repo/tests/test_stage78_prior_guided.py
from __future__ import annotations
import os, glob, asyncio

from engine.config import load_config, save_config
from user_model.policy import set_profile
from grounded.provenance import STORE
from engine.synthesis_pipeline import run_pipeline
from engine.learn_store import LEARN_DIR, load_baseline, _task_key

def _reset():
    os.makedirs(STORE, exist_ok=True)
    os.makedirs(LEARN_DIR, exist_ok=True)
    # נקה CAS + learn
    for p in glob.glob(os.path.join(STORE, "*.json")):
        os.remove(p)
    for p in glob.glob(os.path.join(LEARN_DIR, "*")):
        os.remove(p)

async def _many_runs(spec_name: str, goal: str, n: int = 10):
    cfg = load_config()
    cfg.setdefault("evidence", {}).update({"required": True})
    cfg.setdefault("guard", {}).update({"min_trust": 0.7, "max_age_s": 3600})
    cfg.setdefault("phi", {}).update({"max_allowed": 50_000.0})
    save_config(cfg)
    set_profile("u_prior", min_trust=0.7, max_age_s=3600, strict_grounded=True)

    spec = {"name": spec_name, "goal": goal}
    for _ in range(n):
        out = await run_pipeline(spec, user_id="u_prior", learn=True)
        assert isinstance(out, dict) and "text" in out and "claims" in out

async def test_prior_guides_generation_and_preserves_baseline():
    _reset()

    # 1) צור Baseline התחלתי (הגנרטור הרגיל יבחר A המהירה)
    name, goal = "prior_app", "Hello prior!"
    await _many_runs(name, goal, n=12)

    key = _task_key(name, goal)
    base = load_baseline(key)
    assert base is not None and base["label"] == "A"
    phi0 = float(base["phi"])

    # 2) כעת יש Baseline → run_pipeline ישתמש ב-prior וייצר וריאציות בדמות המנצח
    out = await run_pipeline({"name": name, "goal": goal}, user_id="u_prior", learn=True)
    assert isinstance(out, dict) and "text" in out
    assert "VARIANT=A" in out["text"], "עם prior אמור לבחור ב-A (או A-like תחת אותה תווית)"

    # 3) ודא שאין רגרסיה של Baseline
    base2 = load_baseline(key)
    assert base2 is not None
    assert float(base2["phi"]) <= phi0 + 1e-9

def run():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_prior_guides_generation_and_preserves_baseline())
    print("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())