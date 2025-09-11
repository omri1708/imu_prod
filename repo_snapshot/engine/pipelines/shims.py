# engine/pipelines/shims.py
from __future__ import annotations
import os, json, time
from typing import Any, Dict, List

# ---- ייבוא מהקוד שלך:
from engine.pipeline import Engine as VMEngine                  # (6)
from engine.pipeline_events import run_pipeline_spec            # (3)
from engine.pipeline_multi import run_pipeline_multi            # (4)
# שלוש גרסאות synthesis_pipeline: נגדיר עטיפות בטוחות:
# (8) גרסת A/B async שמקבלת dict
from engine.synthesis_pipeline_ph105 import run_pipeline as ab_run_async   # זהה בשם, אבל async ומקבל dict
# (9) גרסת BuildSpec מלאה
from engine.synthesis_pipeline import SynthesisPipeline as buildspec_run  # זהה בשם, אבל מקבל BuildSpec

# שים לב: כיון ששלוש הגרסאות חולקות שם קובץ, לעיתים תצטרך לייבא לפי נתיב ייחודי
# או לפצל לקבצים שונים. כאן אנו מניחים שהן זמינות כפי ששיתפת.

# ---------- VM ----------
async def run_vm(program: List[Dict[str,Any]], ctx: Dict[str,Any]) -> Dict[str,Any]:
    eng = VMEngine()
    code, body = eng.run_program(program, payload=ctx.get("payload", {}), ctx=ctx)
    return {"ok": 200 <= int(code) < 400, "status": code, "text": body if isinstance(body, str) else None, "body": body}

# ---------- Events JSON spec ----------
async def run_events_spec(spec_text: str, ctx: Dict[str,Any]) -> Dict[str,Any]:
    user = ctx.get("user_id","anon")
    policy = ctx.get("__policy__")
    ev_index = ctx.get("__ev_index__")
    run_id = run_pipeline_spec(user=user, spec_text=spec_text, policy=policy, ev_index=ev_index)
    # pipeline_events כבר אוכף חוזים/ראיות; כאן רק מחזירים מזהה ריצה
    return {"ok": True, "run_id": run_id}

# ---------- A/B Explore (async, dict{name,goal}) ----------
async def run_ab_explore(spec: Dict[str,Any], ctx: Dict[str,Any]) -> Dict[str,Any]:
    user = ctx.get("user_id","anon")
    out = await ab_run_async(spec, user_id=user, learn=bool(ctx.get("learn", False)),
                             domain=ctx.get("domain"), risk_hint=ctx.get("risk_hint"))
    # out כבר כולל guard/gate/pkg/text; נשאיר מבנה דומה
    return dict(out or {}, ok=True)

# ---------- BuildSpec מלא ----------
async def run_buildspec(spec: Any, ctx: Dict[str,Any]) -> Dict[str,Any]:
    user = ctx.get("user_id","anon")
    out = buildspec_run.run(spec, out_root=ctx.get("out_root") or "/mnt/data/imu_builds", user_id=user)
    return dict(out or {}, ok=True)

# ---------- BuildSpec Multi (אם יש פיצול) ----------
async def run_buildspec_multi(spec: Any, ctx: Dict[str,Any]) -> Dict[str,Any]:
    try:
        # אם split_spec(spec) יחזיר יותר מאחד → נריץ multi; אחרת נחזיר buildspec רגיל
        from engine.micro_split import split_spec
        parts = split_spec(spec)
        if not parts or len(parts) <= 1:
            return await run_buildspec(spec, ctx)
        out = run_pipeline_multi(spec, out_root=ctx.get("out_root") or "/mnt/data/imu_builds",
                                 user_id=ctx.get("user_id","anon"))
        return dict(out or {}, ok=True, multi=True)
    except Exception:
        # אם אין micro_split – פשוט ריצה בודדת
        return await run_buildspec(spec, ctx)
