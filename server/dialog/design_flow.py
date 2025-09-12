# server/dialog/design_flow.py
from __future__ import annotations
import json
from typing import Dict, Any
from engine.llm_gateway import LLMGateway
from engine.prompt_builder import PromptBuilder

PB = PromptBuilder()
GW = LLMGateway()

def design_arch(uid: str, spec: Dict[str,Any], ctx: Dict[str,Any]) -> Dict[str,Any]:
    persona = (ctx.get("persona") or {})
    content = {
        "spec_json": json.dumps(spec, ensure_ascii=False),
        "context_json": json.dumps(ctx, ensure_ascii=False)
    }
    prompt = PB.compose(uid, "design", "arch", persona, content, json_only=True)
    r = GW.structured(uid, task="planning", intent="build_architecture",
                      schema_hint=None, prompt=f"{prompt['system']}\n{prompt['user']}",
                      require_grounding=False, temperature=0.1)
    return r.get("json") or {}
