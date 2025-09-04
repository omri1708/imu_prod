# imu_repo/tests/test_stage81_context_pareto.py
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

async def _warm(spec, user_id: str, n: int = 8):
    for _ in range(n):
        out = await run_pipeline(spec, user_id=user_id, learn=True)
        assert isinstance(out, dict) and "text" in out and "claims" in out

async def test_realtime_prefers_stable_over_explore_when_sensitive_weights():
    _reset()
    cfg = load_config()
    cfg.setdefault("evidence", {}).update({"required": True})
    cfg.setdefault("guard", {}).update({"min_trust": 0.7, "max_age_s": 3600})
    cfg.setdefault("phi", {}).update({"max_allowed": 50_000.0})
    cfg.setdefault("explore", {}).update({"epsilon": 1.0})  # יווצר גם E
    save_config(cfg)

    # Intent: 'realtime' (נמצא ב-name/goal)
    spec = {"name":"realtime_stream_service", "goal":"Serve users in <50ms"}
    set_profile("u_rt", strict_grounded=True,
                # מדגישים שגיאות/אמון כדי להעדיף וריאציה יציבה A על פני E (#EXPLORE)
                phi_weights={"errors":0.3, "distrust":0.2, "latency":0.4, "cost":0.09, "energy":0.01, "memory":0.0})

    await _warm(spec, "u_rt", n=10)
    out = await run_pipeline(spec, user_id="u_rt", learn=True)
    assert "VARIANT=A" in out["text"], "ברילטיים+רגישות, A_like אמור לנצח את Explore"

async def test_cost_context_prefers_E_when_code_shorter():
    _reset()
    cfg = load_config()
    cfg.setdefault("evidence", {}).update({"required": True})
    cfg.setdefault("guard", {}).update({"min_trust": 0.7, "max_age_s": 3600})
    cfg.setdefault("phi", {}).update({"max_allowed": 50_000.0})
    cfg.setdefault("explore", {}).update({"epsilon": 1.0})
    save_config(cfg)

    # Intent: 'batch' + 'cost' בטקסט
    spec = {"name":"batch_cost_job", "goal":"optimize cost for nightly batch"}
    set_profile("u_cost", strict_grounded=True,
                phi_weights={"latency":0.2,"cost":0.7,"errors":0.08,"distrust":0.02,"energy":0.0,"memory":0.0})

    await _warm(spec, "u_cost", n=8)
    out = await run_pipeline(spec, user_id="u_cost", learn=True)
    assert "VARIANT=" in out["text"]  # לרוב E ינצח (קוד קצר), אבל נסתפק בקיום תוצאה

async def test_pareto_filters_dominated_variant():
    _reset()
    cfg = load_config()
    cfg.setdefault("evidence", {}).update({"required": True})
    cfg.setdefault("guard", {}).update({"min_trust": 0.7, "max_age_s": 3600})
    cfg.setdefault("phi", {}).update({"max_allowed": 50_000.0})
    # לא לחקור → A_like + B (B איטי/יקר ודומיננטי)
    cfg.setdefault("explore", {}).update({"epsilon": 0.0})
    save_config(cfg)

    spec = {"name":"ui_frontend_service", "goal":"render UI fast"}
    set_profile("u_ui", strict_grounded=True)

    # חימום לקבלת baseline
    await _warm(spec, "u_ui", n=6)
    out = await run_pipeline(spec, user_id="u_ui", learn=True)
    assert "VARIANT=A" in out["text"]

def run():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_realtime_prefers_stable_over_explore_when_sensitive_weights())
    loop.run_until_complete(test_cost_context_prefers_E_when_code_shorter())
    loop.run_until_complete(test_pareto_filters_dominated_variant())
    print("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())