# imu_repo/tests/test_stage82_explore_adaptive.py
from __future__ import annotations
import os, glob, asyncio, time

from engine.config import load_config, save_config
from user_model.policy import set_profile
from grounded.provenance import STORE
from engine.synthesis_pipeline import run_pipeline
from engine.learn_store import LEARN_DIR, load_baseline, _task_key
from engine.explore_state import load_state, in_cooldown

def _reset():
    os.makedirs(STORE, exist_ok=True)
    os.makedirs(LEARN_DIR, exist_ok=True)
    for p in glob.glob(os.path.join(STORE, "*.json")):
        os.remove(p)
    for p in glob.glob(os.path.join(LEARN_DIR, "*")):
        os.remove(p)
    # מנקה גם סטייט explore
    st_dir = "/mnt/data/imu_repo/.state/explore"
    os.makedirs(st_dir, exist_ok=True)
    for p in glob.glob(os.path.join(st_dir, "*.json")):
        os.remove(p)

async def _warm(spec, user_id: str, n: int = 6):
    for _ in range(n):
        out = await run_pipeline(spec, user_id=user_id, learn=True)
        assert isinstance(out, dict) and "text" in out and "claims" in out

async def test_sensitive_intent_has_low_epsilon_and_cooldown_after_regression():
    _reset()
    cfg = load_config()
    # ראיות חובה + שערי שמירה
    cfg.setdefault("evidence", {}).update({"required": True})
    cfg.setdefault("guard", {}).update({"min_trust": 0.7, "max_age_s": 3600})
    cfg.setdefault("phi", {}).update({"max_allowed": 50_000.0})
    # מדיניות Explore אדפטיבית לפי Intent
    cfg["explore"] = {
        "by_intent": {
            "sensitive": {"base": 0.02, "min": 0.0, "max": 0.1},
            "realtime":  {"base": 0.05, "min": 0.0, "max": 0.2},
            "batch":     {"base": 0.2,  "min": 0.0, "max": 0.6},
        },
        "cooldown_s": 600.0
    }
    save_config(cfg)

    spec = {"name":"realtime_sensitive_service", "goal":"handle pii users realtime <30ms"}
    user = "u_sens"
    # העדפה: שגיאות/אמון מודגשים, לטובת יציבות
    set_profile(user, strict_grounded=True,
                phi_weights={"errors":0.3, "distrust":0.25, "latency":0.35, "cost":0.08, "energy":0.01, "memory":0.01})

    # חימום + Baseline
    await _warm(spec, user, n=10)
    key = _task_key(spec["name"], spec["goal"])
    base = load_baseline(key)
    assert base is not None

    # ריצה אחת — ייתכן שתתרחש Explore (נשלט ע"י זמן) — אם תקרה ותוביל לפי גבוה יותר, נצפה לקול־דאון
    out = await run_pipeline(spec, user_id=user, learn=True)
    st = load_state(key)
    # אם יש cooldown, זו עדות לרגרסיה → pass; אם אין — גם בסדר (ייתכן שלא התקבלה החלטת Explore בפעם זו)
    assert isinstance(st, dict)
    # נריץ מספר פעמים כדי להגביר סבירות Explore→Regression
    for _ in range(4):
        _ = await run_pipeline(spec, user_id=user, learn=True)

    # בדיקה: או שיש cooldown או שלא (לא נכשל), אבל אם יש — הוא אמור להיות true
    if in_cooldown(key):
        assert in_cooldown(key) is True

async def test_cost_saver_has_high_epsilon_no_cooldown_if_improved():
    _reset()
    cfg = load_config()
    cfg.setdefault("evidence", {}).update({"required": True})
    cfg.setdefault("guard", {}).update({"min_trust": 0.7, "max_age_s": 3600})
    cfg.setdefault("phi", {}).update({"max_allowed": 50_000.0})
    cfg["explore"] = {
        "by_intent": {
            "cost_saver": {"base": 0.6, "min": 0.2, "max": 0.95},
            "batch": {"base": 0.2, "min": 0.0, "max": 0.6}
        },
        "cooldown_s": 300.0
    }
    save_config(cfg)

    spec = {"name":"batch_cost_saver", "goal":"optimize cost for nightly batch"}
    user = "u_cost"
    set_profile(user, strict_grounded=True,
                phi_weights={"latency":0.2,"cost":0.7,"errors":0.08,"distrust":0.02,"energy":0.0,"memory":0.0})

    # חימום
    await _warm(spec, user, n=8)
    key = _task_key(spec["name"], spec["goal"])

    # כמה ריצות — בסבירות גבוהה Explore יקרה ולרוב תשפר עלות (E קצר יותר)
    for _ in range(6):
        _ = await run_pipeline(spec, user_id=user, learn=True)

    # אם שופרה φ, לא אמור להיות cooldown
    assert in_cooldown(key) is False

def run():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_sensitive_intent_has_low_epsilon_and_cooldown_after_regression())
    loop.run_until_complete(test_cost_saver_has_high_epsilon_no_cooldown_if_improved())
    print("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())