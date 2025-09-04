# imu_repo/tests/test_stage84_guard_enforce.py
from __future__ import annotations
import os, glob, asyncio, time

from grounded.provenance import STORE
from grounded.claims import current
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
    current().reset()

def _set_cfg(required: bool, min_trust: float, max_age_s: float | None, min_count: int = 1, required_kinds=None):
    cfg = load_config()
    cfg["evidence"] = {"required": required, "signing_secret": "stage84_secret"}
    guard = {"min_trust": min_trust}
    if max_age_s is not None:
        guard["max_age_s"] = float(max_age_s)
    guard["min_count"] = int(min_count)
    if required_kinds is not None:
        guard["required_kinds"] = list(required_kinds)
    cfg["guard"] = guard
    cfg["phi"] = {"max_allowed": 50_000.0}
    save_config(cfg)

async def test_guard_rejects_when_trust_too_high_and_falls_back():
    _reset_all()
    # קונפיג קשוח מדי — min_trust=0.99 כך שרוב הראיות (0.95) ייפסלו
    _set_cfg(required=True, min_trust=0.99, max_age_s=3600.0, min_count=2, required_kinds=["spec","plan"])
    spec = {"name":"rt_service", "goal":"respond < 40ms under guard"}
    set_profile("u", strict_grounded=True, phi_weights={"errors":0.3,"distrust":0.2,"latency":0.4,"cost":0.08,"energy":0.01,"memory":0.01})
    out = await run_pipeline(spec, user_id="u", learn=True)
    assert "text" in out and "[FALLBACK]" in out["text"]
    assert out.get("guard_rejected", False) is True
    # יש ראיית fallback
    evs = current().snapshot()
    kinds = {e.get("kind") for e in evs}
    assert "fallback_used" in kinds

async def test_guard_allows_when_requirements_met():
    _reset_all()
    # רף סביר — min_trust=0.7, min_count=2, ומספקים kinds שנוצרו בצנרת
    _set_cfg(required=True, min_trust=0.7, max_age_s=3600.0, min_count=2, required_kinds=["spec","plan"])
    spec = {"name":"batch_job", "goal":"nightly optimization"}
    set_profile("u2", strict_grounded=True, phi_weights={"cost":0.6,"latency":0.2,"errors":0.19,"distrust":0.01,"energy":0.0,"memory":0.0})
    out = await run_pipeline(spec, user_id="u2", learn=True)
    assert "text" in out and "[FALLBACK]" not in out["text"]
    # ויש לפחות 2 ראיות מהסוגים הנדרשים
    evs = current().snapshot()
    kinds = [e.get("kind") for e in evs]
    assert all(k in kinds for k in ["spec","plan"])

async def test_guard_rejects_on_staleness():
    _reset_all()
    # נעשה evidence טריות ואז "נצניף" את הזמן ע"י max_age_s קטן מאוד
    _set_cfg(required=True, min_trust=0.7, max_age_s=0.001, min_count=1, required_kinds=["spec"])
    spec = {"name":"gpu_pipeline", "goal":"train within budget"}
    set_profile("u3", strict_grounded=True, phi_weights={"errors":0.2,"distrust":0.1,"latency":0.4,"cost":0.2,"energy":0.1})
    # מייד אחרי הריצה — עדיין ייתכן שלא ייחשב כמיושן; נחכה מעט
    out = await run_pipeline(spec, user_id="u3", learn=True)
    # השהיה קלה כדי לחרוג מה-max_age_s
    time.sleep(0.01)
    out2 = await run_pipeline(spec, user_id="u3", learn=True)
    # לפחות אחת מהריצות אמורה להיפסל על רקע סטייל
    assert ("[FALLBACK]" in out["text"]) or ("[FALLBACK]" in out2["text"])

def run():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_guard_rejects_when_trust_too_high_and_falls_back())
    loop.run_until_complete(test_guard_allows_when_requirements_met())
    loop.run_until_complete(test_guard_rejects_on_staleness())
    print("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())