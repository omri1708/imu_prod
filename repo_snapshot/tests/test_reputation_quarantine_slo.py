# imu_repo/tests/test_reputation_quarantine_slo.py
from __future__ import annotations
import time
import os
import types

from engine.synthesis_pipeline import SynthesisPipeline, PipelineError
from engine.quarantine import Quarantined
from engine.reputation import ReputationRegistry

def _fake_now_gen(start: float = 1_700_000_000.0):
    t = {"now": start}
    def now():
        return t["now"]
    def advance(sec: float):
        t["now"] += sec
    return now, advance

def test_reputation_influences_trust_indirectly():
    # מדגים שנבנה רישום reputation ושאפשר לעדכן אותו; הטרסט עצמו נצרך ב-Gate אחר במערכת
    rep = ReputationRegistry(half_life_days=1.0, alpha=0.5)
    base = rep.factor("example.com")
    assert 0.5 <= base <= 1.5
    rep.update_on_success("example.com", 0.4)
    f2 = rep.factor("example.com")
    assert f2 >= base  # פקטור עלה

def test_quarantine_triggers_and_releases():
    now, advance = _fake_now_gen()
    pol = {
        "quarantine_min_calls": 5,
        "quarantine_error_rate_threshold": 0.4,  # 40%
        "quarantine_violation_rate_threshold": 0.2,
        "quarantine_backoff_base_sec": 60.0,
        "quarantine_backoff_max_sec": 600.0
    }
    pipe = SynthesisPipeline(policy=pol, now=now)

    # שלב 'test' שמחזיר כישלון לרוב
    def test_step(ctx):
        calls = ctx.setdefault("_calls", 0) + 1
        ctx["_calls"] = calls
        ok = (calls % 3 == 0)  # 1,2,3 -> false,false,true -> 66% כשלון
        return {"ok": ok}

    pipe.register("test", test_step)

    # 5 קריאות — אמור להיכנס להסגר
    quarantined = False
    for _ in range(5):
        try:
            pipe.run({})
        except PipelineError:
            pass
        except Exception:
            pass
    # עכשיו ההסגר פעיל: נסיון נוסף צריך להיזרק
    try:
        pipe.run({})
        assert False, "should be quarantined"
    except Quarantined:
        quarantined = True
    assert quarantined

    # קידום זמן עד שחרור
    advance(61.0)
    # עכשיו מותר שוב
    try:
        pipe.run({})
    except Quarantined:
        assert False, "should have been released from quarantine"

def test_slo_p95_gate_blocks_rollout():
    now, advance = _fake_now_gen()
    pol = {"rollout_p95_ms": 50.0, "p95_window": 50}
    pipe = SynthesisPipeline(policy=pol, now=now)

    def rollout_step(ctx):
        # מדמה זמן ריצה ארוך
        time.sleep(0.06)  # ~60ms
        return {"ok": True}

    pipe.register("rollout", rollout_step)

    # מספר ריצות כדי למלא חלון
    blocked = False
    for _ in range(5):
        try:
            pipe.run({})
        except PipelineError as e:
            assert "slo_breach:rollout" in str(e)
            blocked = True
            break
    assert blocked, "p95 gate should have blocked rollout"