# imu_repo/tests/test_stage73_end2end_claims_everywhere.py
from __future__ import annotations
import asyncio, os
from typing import Dict, Any, List, Union

from engine.config import load_config, save_config
from user_model.policy import set_profile
from realtime.ws_server import WSServer
from engine.hooks import AsyncThrottle, ThrottleConfig
from alerts.notifier import metrics_log, ROOT as LOG_ROOT

def _reset_logs():
    os.makedirs(LOG_ROOT, exist_ok=True)
    for fn in ("metrics.jsonl","alerts.jsonl","provenance.jsonl"):
        p = os.path.join(LOG_ROOT, fn)
        if os.path.exists(p): os.remove(p)

async def _reverse(s: str) -> str:
    return s[::-1]

async def test_all_responses_have_claims():
    # אוכפים Evidences חובה
    cfg = load_config()
    cfg.setdefault("evidence", {}).update({"required": True})
    cfg.setdefault("guard", {}).update({"min_trust": 0.7, "max_age_s": 3600})
    save_config(cfg)

    # פרופיל משתמש (מחמיר/מקֵל לפי הצורך — כאן ברירת מחדל סבירה)
    set_profile("u_end2end", min_trust=0.7, max_age_s=3600, strict_grounded=True)

    s = WSServer(handler=_reverse, chunk_size=64000, permessage_deflate=True)
    s.attach_throttle(AsyncThrottle(ThrottleConfig(capacity=6, refill_per_sec=100.0)))
    await s.bind_user("u_end2end")   # כל תשובה מכאן והלאה חייבת claims

    # "תנועה" — וגם נרשום מדדים לצורך ניטור
    async def call_one(i: int):
        msg = f"m-{i}"
        out = await s.handle(msg)
        # מאחר והוגדר bind_user — out הוא dict עם text/claims
        assert isinstance(out, dict), f"expected dict, got {type(out)}"
        assert out.get("text") == msg[::-1]
        cl = out.get("claims")
        assert isinstance(cl, list) and len(cl) > 0
        # מדד זמן לוגי — כאן אין אמת־מידה אמיתית של רשת; נרשום מדד קל
        metrics_log("guarded_handler", {"ok": True, "latency_ms": 20.0})

    await asyncio.gather(*[asyncio.create_task(call_one(i)) for i in range(100)])
    s.close()

def run():
    _reset_logs()
    asyncio.get_event_loop().run_until_complete(test_all_responses_have_claims())
    print("OK")
    return 0

if __name__=="__main__":
    raise SystemExit(run())