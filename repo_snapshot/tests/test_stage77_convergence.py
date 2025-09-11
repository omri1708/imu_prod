# imu_repo/tests/test_stage77_convergence.py
from __future__ import annotations
import os, glob, asyncio, json

from engine.config import load_config, save_config
from user_model.policy import set_profile
from grounded.provenance import STORE
from engine.synthesis_pipeline import run_pipeline
from engine.learn_store import LEARN_DIR, load_baseline, _task_key
from synth import generate_ab as gen_ab_module

def _reset():
    os.makedirs(STORE, exist_ok=True)
    os.makedirs(LEARN_DIR, exist_ok=True)
    # נקה CAS + learn
    for p in glob.glob(os.path.join(STORE, "*.json")):
        os.remove(p)
    for p in glob.glob(os.path.join(LEARN_DIR, "*")):
        os.remove(p)

async def _many_runs(spec_name: str, goal: str, n: int = 25):
    cfg = load_config()
    cfg.setdefault("evidence", {}).update({"required": True})
    cfg.setdefault("guard", {}).update({"min_trust": 0.7, "max_age_s": 3600})
    cfg.setdefault("phi", {}).update({"max_allowed": 50_000.0})
    save_config(cfg)
    set_profile("u_conv", min_trust=0.7, max_age_s=3600, strict_grounded=True)

    spec = {"name": spec_name, "goal": goal}
    for _ in range(n):
        out = await run_pipeline(spec, user_id="u_conv", learn=True)
        assert isinstance(out, dict) and "text" in out and "claims" in out

def _baseline_of(spec_name: str, goal: str):
    key = _task_key(spec_name, goal)
    return load_baseline(key)

async def test_convergence_and_regression_guard():
    _reset()

    # 1) הפעל הרבה ריצות — A אמור לנצח (מהיר), Baseline יאומץ ויתכנס
    name, goal = "conv_app", "Hello convergence!"
    await _many_runs(name, goal, n=25)
    base1 = _baseline_of(name, goal)
    assert base1 is not None, "baseline should exist"
    phi1 = float(base1["phi"])
    assert base1["label"] == "A"  # הוריאציה המהירה

    # 2) נסה 'רגרסיה': תן מחולל וריאציות שמחזיר אופציות איטיות בלבד
    saved = gen_ab_module.generate_variants
    def slow_variants(_spec):
        return [
            {"label":"C","language":"python","code":"#SLOW\ndef main():\n    return 'slow'"},
            {"label":"D","language":"python","code":"#SLOW\ndef main():\n    return 'slower'"},
        ]
    gen_ab_module.generate_variants = slow_variants

    # עדיין נריץ עם learn=True — ה־learn אמור "להחזיק" ולא לקדם Baseline (Regression Guard)
    await _many_runs(name, goal, n=3)
    base2 = _baseline_of(name, goal)
    assert base2 is not None
    # אסור שה־phi יגדל / יוחלף ל"גרוע יותר"
    assert float(base2["phi"]) <= phi1 + 1e-9
    assert base2["label"] == "A"  # עדיין A

    # השב את המחולל
    gen_ab_module.generate_variants = saved

def run():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_convergence_and_regression_guard())
    print("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())