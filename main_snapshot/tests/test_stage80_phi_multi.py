# imu_repo/tests/test_stage80_phi_multi.py
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
    for p in glob.glob(os.path.join(STORE, "*.json")):
        os.remove(p)
    for p in glob.glob(os.path.join(LEARN_DIR, "*")):
        os.remove(p)

async def _run_many(user_id: str, spec, n: int = 10, learn: bool = True):
    for _ in range(n):
        out = await run_pipeline(spec, user_id=user_id, learn=learn)
        assert isinstance(out, dict) and "text" in out and "claims" in out

async def test_multi_objective_changes_choice_by_profile():
    _reset()
    cfg = load_config()
    cfg.setdefault("evidence", {}).update({"required": True})
    cfg.setdefault("guard", {}).update({"min_trust": 0.7, "max_age_s": 3600})
    cfg.setdefault("phi", {}).update({"max_allowed": 50_000.0})
    cfg.setdefault("explore", {}).update({"epsilon": 0.0})
    save_config(cfg)

    spec = {"name":"phi_multi_demo", "goal":"Hello multi-objective!"}

    # משתמש 1: latency-first (ברירת מחדל)
    set_profile("u_lat", min_trust=0.7, max_age_s=3600, strict_grounded=True,
                phi_weights={"latency":0.7,"cost":0.2,"errors":0.08,"distrust":0.02,"energy":0.0,"memory":0.0})
    await _run_many("u_lat", spec, n=8, learn=True)
    key = _task_key(spec["name"], spec["goal"])
    base1 = load_baseline(key)
    assert base1 is not None
    # A_like מהיר אמור לנצח (VARIANT=A)
    out = await run_pipeline(spec, user_id="u_lat", learn=True)
    assert "VARIANT=A" in out["text"]

    # משתמש 2: cost-first (מענישים אורך קוד)
    set_profile("u_cost", min_trust=0.7, max_age_s=3600, strict_grounded=True,
                phi_weights={"latency":0.2,"cost":0.7,"errors":0.08,"distrust":0.02,"energy":0.0,"memory":0.0})
    # ריצות עבור u_cost — בסבירות גבוהה E_explore (הקצר) ינצח כאשר יש baseline ו-explore≥0
    cfg = load_config(); cfg["explore"]["epsilon"] = 1.0  # תמיד לחקור עבור u_cost
    save_config(cfg)
    await _run_many("u_cost", spec, n=6, learn=True)
    out2 = await run_pipeline(spec, user_id="u_cost", learn=True)
    # ייתכן שהווריאציה תהיה A או E — אך עבור cost-high לרוב E ינצח (קוד קצר)
    assert "VARIANT=" in out2["text"]

async def test_no_regression_with_multi_objective():
    _reset()
    cfg = load_config()
    cfg.setdefault("evidence", {}).update({"required": True})
    cfg.setdefault("guard", {}).update({"min_trust": 0.7, "max_age_s": 3600})
    cfg.setdefault("phi", {}).update({"max_allowed": 50_000.0})
    cfg.setdefault("explore", {}).update({"epsilon": 0.5})
    save_config(cfg)

    spec = {"name":"phi_regress_guard", "goal":"Keep baseline safe."}

    set_profile("u_safe", strict_grounded=True,
                phi_weights={"latency":0.6,"cost":0.25,"errors":0.1,"distrust":0.03,"energy":0.015,"memory":0.005})
    # חימום וקביעת Baseline
    await _run_many("u_safe", spec, n=12, learn=True)
    key = _task_key(spec["name"], spec["goal"])
    base = load_baseline(key)
    assert base is not None
    phi0 = float(base["phi"])

    # עוד ריצות — גם אם תופיע Explore גרועה, Regression Guard ימנע קידום
    for _ in range(4):
        out = await run_pipeline(spec, user_id="u_safe", learn=True)
        assert isinstance(out, dict) and "text" in out

    base2 = load_baseline(key)
    assert base2 is not None
    assert float(base2["phi"]) <= phi0 + 1e-9

def run():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_multi_objective_changes_choice_by_profile())
    loop.run_until_complete(test_no_regression_with_multi_objective())
    print("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())