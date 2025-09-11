# imu_repo/tests/test_stage72_user_grounded_defaults.py
from __future__ import annotations
import asyncio, time
from typing import Any, Dict

from engine.config import load_config, save_config
from user_model.policy import set_profile, resolve_gate
from engine.pipeline_defaults import build_user_guarded

async def _calc_hex(x: int) -> str:
    # חישוב דטרמיניסטי "טהור"
    return f"{x:x}"

async def test_pass_with_user_policy():
    # evidence.required=True כדי לאכוף claims
    cfg = load_config()
    cfg.setdefault("evidence", {}).update({"required": True})
    save_config(cfg)

    # פרופיל משתמש: min_trust נמוך מ-1.0 כדי שהראיה האוטומטית (trust=1.0) תעבור
    set_profile("u1", min_trust=0.9, max_age_s=3600, strict_grounded=True)
    gate = resolve_gate("u1")
    assert gate["min_trust"] == 0.9

    wrapped = await build_user_guarded(_calc_hex, user_id="u1")
    out = await wrapped(255)
    assert out["text"] == "ff"
    assert out.get("claims"), "strict mode must guarantee claims"

async def test_fail_with_too_strict_age():
    cfg = load_config()
    cfg.setdefault("evidence", {}).update({"required": True})
    save_config(cfg)

    # כעת נדרוש max_age_s קטן יותר מן ה-"ttl_s" של הראיה האוטומטית (5)
    set_profile("u2", min_trust=0.5, max_age_s=1, strict_grounded=True)
    wrapped = await build_user_guarded(_calc_hex, user_id="u2")
    failed=False
    try:
        await wrapped(15)
    except PermissionError:
        failed=True
    assert failed, "should fail when user policy's max_age_s < auto ttl_s"

def run():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_pass_with_user_policy())
    loop.run_until_complete(test_fail_with_too_strict_age())
    print("OK")
    return 0

if __name__=="__main__":
    raise SystemExit(run())