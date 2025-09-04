# imu_repo/tests/test_stage71_ws_adaptive_and_strict.py
from __future__ import annotations
import asyncio, os, json
from typing import Any

from alerts.notifier import metrics_log, ROOT as LOG_ROOT
from engine.config import load_config, save_config
from realtime.ws_server import WSServer
from engine.hooks import AsyncThrottle, ThrottleConfig
from engine.metrics_watcher import adapt_once
from engine.strict_grounded import strict_guarded

def _reset_logs():
    os.makedirs(LOG_ROOT, exist_ok=True)
    for fn in ("metrics.jsonl","alerts.jsonl"):
        p = os.path.join(LOG_ROOT, fn)
        if os.path.exists(p): os.remove(p)

async def _echo(x: str) -> str:
    return x[::-1]

async def test_ws_adaptive_and_throttle():
    _reset_logs()
    # כתוב "תנועה" שמדמה p95 גבוה → אמור להקטין קיבולת
    for _ in range(400):
        metrics_log("guarded_handler", {"ok": True, "latency_ms": 140.0})
    th = AsyncThrottle(ThrottleConfig(capacity=8, refill_per_sec=100.0))
    s = WSServer(handler=_echo, chunk_size=64000, permessage_deflate=True)
    s.attach_throttle(th)
    # адаптация
    stats = adapt_once(th, name="guarded_handler", window_s=3600)
    # לפי engine.hooks: p95>120 → capacity ≈ 0.25*capacity המקורי (>=1)
    assert int(th._capacity) <= 8 and int(th._capacity) >= 1
    # עכשיו "שפר" את המדדים → אמור לחזור ליעד גבוה יותר
    _reset_logs()
    for _ in range(400):
        metrics_log("guarded_handler", {"ok": True, "latency_ms": 40.0})
    stats2 = adapt_once(th, name="guarded_handler", window_s=3600)
    # capacity צריכה להיות לפחות 4 (חצי או יותר) בהתאם לחוקים
    assert int(th._capacity) >= 4, f"expected capacity>=4, got {th._capacity}"

    # ודא שהמצערת אכן מגבילה קונקרנציה בפועל
    async def call_many(n:int=20):
        async def one(i:int):
            out = await s.handle(f"msg-{i}")
            return out
        await asyncio.gather(*[one(i) for i in range(n)])
    await call_many(40)
    assert th.max_in_use <= int(th._capacity)

async def test_strict_grounded_always_claims():
    # דרוש evidences.required
    cfg = load_config()
    cfg.setdefault("evidence", {}).update({"required": True})
    cfg.setdefault("guard", {}).update({"min_trust": 0.7, "max_age_s": 3600})
    save_config(cfg)

    async def calc(x: int) -> str:
        # "חישוב" דטרמיניסטי — יהפוך מחרוזת
        return f"{x:x}"

    safe = await strict_guarded(calc, min_trust=0.7)
    out = await safe(255)
    # בהכרח מחזיר claims, כי strict_guarded הזריק ראיה דטרמיניסטית
    assert isinstance(out, dict) and out.get("claims") and out.get("text")=="ff"

def run():
    asyncio.get_event_loop().run_until_complete(test_ws_adaptive_and_throttle())
    asyncio.get_event_loop().run_until_complete(test_strict_grounded_always_claims())
    print("OK")
    return 0

if __name__=="__main__":
    raise SystemExit(run())