# server/dialog/intent_router.py
from __future__ import annotations
import json
from typing import Dict, Any, List
from engine.llm_gateway import LLMGateway
from engine.prompt_builder import PromptBuilder

PB = PromptBuilder()
GW = LLMGateway()

def classify_intent(uid: str, msg: str, ctx: Dict[str,Any]) -> Dict[str,Any]:
    """
    מחזיר: {"intent": "...", "confidence": 0..1, "targets": [...], "why": "..."}
    הלוגיקה כולה אצל המודל (messages) – אין heuristics של מילות מפתח.
    """
    schema = {
        "type": "object",
        "properties": {
            "intent": { "type":"string", "enum":["build","clarify","design","ask","other"] },
            "confidence": { "type":"number", "minimum":0, "maximum":1 },
            "targets": { "type":"array", "items": { "type":"string" } },
            "why": { "type":"string" }
        },
        "required": ["intent","confidence"]
    }
    content = {
        "user_text": msg,
        "context_json": json.dumps(ctx, ensure_ascii=False),
        "policy": (
            "Choose intent=build only when the user explicitly asks to build now, "
            "and there is enough info to start. If ambiguous or exploratory, choose clarify/design/ask. "
            "Return JSON only."
        )
    }
    res = GW.structured(
        uid, task="router", intent="intent_classify",
        schema_hint=json.dumps(schema, ensure_ascii=False),
        content=content, require_grounding=False, temperature=0.0
    )
    j = (res or {}).get("json") or {}
    out = {
        "intent": str(j.get("intent") or "other"),
        "confidence": float(j.get("confidence") or 0.0),
        "targets": list(j.get("targets") or []),
        "why": str(j.get("why") or "")
    }
    return out

def extract_slots(uid: str, msg: str, required_slots: List[Dict[str,str]]) -> Dict[str,Any]:
    names = [rs["name"] for rs in required_slots if rs.get("name")]
    if not names: return {"values":{}, "missing":[]}
    content = {
        "user_text": msg,
        "required_keys_json": json.dumps(names, ensure_ascii=False),
    }
    r = GW.structured(
        uid, task="router", intent="slot_extract",
        schema_hint=None, content=content,
        require_grounding=False, temperature=0.0,
    )
    vals = (r.get("json") or {}).get("values") or {}
    missing = [n for n in names if n not in vals or not str(vals[n]).strip()]
    return {"values": vals, "missing": missing}
