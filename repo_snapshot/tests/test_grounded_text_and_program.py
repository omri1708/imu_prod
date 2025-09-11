# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, pytest
from assurance.respond_text import GroundedResponder
from program.orchestrator import ProgramOrchestrator
from assurance.errors import ResourceRequired, ValidationFailed

def test_grounded_responder_with_file(tmp_path):
    p = tmp_path/"src.txt"; p.write_text("hello evidence", encoding="utf-8")
    gr = GroundedResponder("./assurance_store_texttest")
    out = gr.respond_from_sources("demo", [{"file": str(p)}])
    assert out["ok"] is True and out["payload"]["citations"]

@pytest.mark.asyncio
async def test_program_orchestrator_python_or_resource_required():
    orch = ProgramOrchestrator("./assurance_store_progtest")
    spec = {"name":"calc","services":[{"type":"python_app","name":"svc1"}]}
    try:
        r = await orch.build("demo-user", spec)
        assert r["ok"] is True and r["payload"]["services"][0]["compile"] is True
    except ResourceRequired:
        # אם אין python/bwrap — המערכת לא “ממציאה” משאב
        assert True
