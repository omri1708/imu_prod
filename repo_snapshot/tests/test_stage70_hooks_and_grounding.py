# imu_repo/tests/test_stage70_hooks_and_grounding.py
from __future__ import annotations
import os, asyncio, json, glob
from typing import Any, Dict, List

from engine.config import load_config, save_config
from engine.hooks import AsyncThrottle, ThrottleConfig
from engine.evidence_middleware import guarded_handler
from grounded.provenance import STORE, verify_signature

def _reset_env():
    # ודא שהחנות קיימת ונקייה (לא מוחקים הסטורית; רק לטסט)
    os.makedirs(STORE, exist_ok=True)

async def _noop(x: str) -> str:
    return f"ok:{x}"

async def test_guard_and_provenance():
    cfg = load_config()
    cfg.setdefault("guard", {}).update({"min_trust": 0.7, "max_age_s": 3600})
    cfg.setdefault("evidence", {}).update({"required": True})
    save_config(cfg)

    # 1) בלי ראיות → חייב להיכשל
    g1 = await guarded_handler(_noop, min_trust=0.7)
    failed=False
    try:
        await g1("a")
    except PermissionError:
        failed=True
    assert failed, "guard must deny when no evidences"

    # 2) הוסף ראיה טובה → יעבור + יישמרו claims חתומים
    # ניסוח ראיות דרך current().add_evidence קיים ב־engine.evidence_middleware (fallback) או ב־grounded.claims
    from engine.evidence_middleware import current
    cur = current()
    cur.add_evidence("t1", {"source_url":"https://example", "trust":0.9, "ttl_s":60, "payload":{"k":"v"}})
    g2 = await guarded_handler(_noop, min_trust=0.7)
    out = await g2("b")
    assert out["text"]=="ok:b"
    assert out.get("claims") and isinstance(out["claims"], list)

    # אימות קיום קבצי CAS וחתימות
    files = sorted(glob.glob(os.path.join(STORE, "*.json")))
    assert files, "expected evidence CAS files"
    # לא יודעים את הסוד כאן; אימות חתימה בוצע בתוך ה־middleware בעזרת assert

async def test_throttle_concurrency():
    # מצערת עם קיבולת 3 — נבדוק שהשימוש השיא לא עובר 3
    th = AsyncThrottle(ThrottleConfig(capacity=3, refill_per_sec=100.0))
    # עבודות "כבדות" קטנות
    async def work(i:int):
        async with th.slot():
            await asyncio.sleep(0.01)
            return i
    tasks = [asyncio.create_task(work(i)) for i in range(10)]
    await asyncio.gather(*tasks)
    assert th.max_in_use <= 3, f"max_in_use={th.max_in_use} should be <= capacity"

def run():
    _reset_env()
    asyncio.get_event_loop().run_until_complete(test_guard_and_provenance())
    asyncio.get_event_loop().run_until_complete(test_throttle_concurrency())
    print("OK")
    return 0

if __name__=="__main__":
    raise SystemExit(run())