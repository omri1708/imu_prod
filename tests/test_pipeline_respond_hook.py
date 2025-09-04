# imu_repo/tests/test_pipeline_respond_hook.py
from __future__ import annotations
import types
from engine.pipeline_respond_hook import pipeline_respond

def test_pipeline_respond_calls_agent_emit(monkeypatch):
    captured = {}
    def fake_agent_emit(*, answer_text, ctx, policy):
        captured["answer_text"] = answer_text
        captured["ctx"] = ctx
        captured["policy"] = policy
        return {"ok": True, "echo": answer_text}
    monkeypatch.setattr("engine.agent_emit.agent_emit_answer", fake_agent_emit, raising=True)

    ctx = {"__policy__": {"min_trust": 0.8}, "user_id": "u1"}
    out = pipeline_respond(ctx=ctx, answer_text="HELLO")
    assert out["ok"] and out["echo"] == "HELLO"
    assert captured["policy"] == {"min_trust": 0.8}
