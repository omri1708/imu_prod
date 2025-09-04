# imu_repo/tests/test_stage79_prior_exploration.py
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

async def _warmup_and_get_baseline(name: str, goal: str, runs: int = 12):
    cfg = load_config()
    cfg.setdefault("evidence", {}).update({"required": True})
    cfg.setdefault("guard", {}).update({"min_trust": 0.7, "max_age_s": 3600})
    cfg.setdefault("phi", {}).update({"max_allowed": 50_000.0})
    # נרצה לראות חקירה — נבחר epsilon 0.5: כל ריצה שניה (דטרמיניסטי)
    cfg.setdefault("explore", {}).update({"epsilon": 0.5})
    save_config(cfg)
    set_profile("u_explore", min_trust=0.7, max_age_s=3600, strict_grounded=True)

    spec = {"name": name, "goal": goal}
    for _ in range(runs):
        out = await run_pipeline(spec, user_id="u_explore", learn=True)
        assert isinstance(out, dict) and "text" in out and "claims" in out

    key = _task_key(name, goal)
    base = load_baseline(key)
    assert base is not None
    return base

async def test_exploration_can_promote_when_better():
    _reset()
    name, goal = "explore_app", "Hello explore!"
    base = await _warmup_and_get_baseline(name, goal, runs=12)
    phi0 = float(base["phi"])

    # עוד כמה ריצות — אמור להופיע E_explore מדי פעם (epsilon=0.5),
    # הוא קצר יותר → cost_units נמוך → Φ משופר → קידום בטוח.
    for _ in range(4):
        out = await run_pipeline({"name":name, "goal":goal}, user_id="u_explore", learn=True)
        assert isinstance(out, dict) and "text" in out

    key = _task_key(name, goal)
    base2 = load_baseline(key)
    assert base2 is not None
    assert float(base2["phi"]) <= phi0 + 1e-9, "Φ לא אמור לעלות"
    # ברוב הסבירות (דטרמיניסטי לפי אורך קוד), ה-E המשופר יקודם או שה-A_like יישאר — בשני המקרים אין רגרסיה.

def run():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_exploration_can_promote_when_better())
    print("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())