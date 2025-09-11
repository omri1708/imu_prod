# imu_repo/tests/test_stage88_async_budget.py
from __future__ import annotations
import asyncio, time

from engine.config import load_config, save_config
from engine.user_scope import user_scope
from grounded.claims import current
from engine.capability_wrap import text_capability_for_user

def _config_for_tests():
    cfg = load_config()
    cfg["phi"] = {"max_allowed": 25.0, "per_capability_cost": {"slow.echo": 10.0, "light.echo": 1.5}}
    cfg["guard"] = {"min_trust": 0.0, "max_age_s": 3600.0, "min_count": 0, "required_kinds": []}
    cfg["async"] = {
        "max_global": 4,
        "per_user": 2,
        "per_capability": {"slow.echo": 2, "light.echo": 2},
        "per_capability_rps": {"slow.echo": {"rps": 100.0, "burst": 100}}  # לא מגביל זמן בריצה, רק סמפור
    }
    save_config(cfg)

async def _slow_echo(payload):
    await asyncio.sleep(0.2)
    return f"OK:{payload.get('msg','')}"

async def _light_echo(payload):
    # כמעט ללא השהיה
    await asyncio.sleep(0.01)
    return f"OK:{payload.get('msg','')}"

def test_concurrency_caps():
    _config_for_tests()
    current().reset()

    with user_scope("alice"):
        wrapped = text_capability_for_user(_slow_echo, user_id="alice", capability_name="slow.echo", cost=10.0)

        async def go():
            async def one(i): return await wrapped({"msg": f"{i}"})
            t0 = time.time()
            # per_user=2 → מתוך 6 בקשות, לפחות ~0.6s (ריצות בגלים של 2)
            tasks = [asyncio.create_task(one(i)) for i in range(6)]
            outs = await asyncio.gather(*tasks)
            t1 = time.time()
            return outs, t1 - t0

        outs, dt = asyncio.get_event_loop().run_until_complete(go())
        # כולן הסתיימו תקין
        assert all("text" in o and o["text"].startswith("OK:") for o in outs)
        assert dt >= 0.55, f"expected concurrency cap to stretch runtime, got {dt:.3f}s"

def test_phi_budget_exhausts_and_fallback():
    _config_for_tests()
    current().reset()

    with user_scope("bob"):
        wrapped = text_capability_for_user(_light_echo, user_id="bob", capability_name="slow.echo", cost=10.0)
        loop = asyncio.get_event_loop()

        o1 = loop.run_until_complete(wrapped({"msg": "1"}))
        o2 = loop.run_until_complete(wrapped({"msg": "2"}))
        o3 = loop.run_until_complete(wrapped({"msg": "3"}))  # צריך למצות תקציב (25) אחרי שתי ריצות (20)

        assert "[FALLBACK]" not in o1["text"]
        assert "[FALLBACK]" not in o2["text"]
        assert "[FALLBACK]" in o3["text"]
        assert o3.get("budget_exceeded") is True

def run():
    test_concurrency_caps()
    test_phi_budget_exhausts_and_fallback()
    print("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())