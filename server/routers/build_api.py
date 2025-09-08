# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
from fastapi import APIRouter, HTTPException

from engine.intent_to_spec import IntentToSpec
from engine.spec_refiner import SpecRefiner
from engine.blueprints.registry import resolve as resolve_blueprint
from engine.build_orchestrator import BuildOrchestrator

router = APIRouter(prefix="/build", tags=["build"])
i2s = IntentToSpec()
ref = SpecRefiner()
bo  = BuildOrchestrator()

@router.post("/from_text")
async def build_from_text(body: Dict[str,Any]):
    uid  = (body.get("user_id") or "user").strip()
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "text required")
    try:
        # 1) Intent -> SPEC
        spec = i2s.from_text(uid, text)
        spec["__source_text__"] = text

        # 2) אם זה custom – משלים entities/behavior בדומיין
        spec = ref.refine_if_needed(uid, spec)

        # 3) בחירת blueprint לפי domain, עם ברירת-מחדל חכמה (generic_backend)
        generator = resolve_blueprint(spec.get("domain","custom"))
        files = generator(spec)

        # 4) Build+Tests+Evidence (py_compile + pytest אם קיים)
        result = await bo.build_python_module(files, name=f"{spec.get('domain','custom')}_backend")
        return {"ok": True, "spec": spec, "build": result}
    except Exception as e:
        raise HTTPException(400, f"failed: {e}")