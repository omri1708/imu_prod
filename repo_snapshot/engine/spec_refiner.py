# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
from engine.llm_gateway import LLMGateway

class SpecRefiner:
    """
    משלים/מחדד SPEC לדומיינים לא-מוכרים:
    - מייצר entities/fields דומייניים
    - מגדיר core_behavior עם נוסחה ומשקלים (ללוגיקה לא-גנרית)
    - מספק סט בדיקות לדוגמה (inputs -> expected)
    הכל דרך LLMGateway במצב JSON בלבד (אין טקסט חופשי).
    """
    def __init__(self):
        self.gw = LLMGateway()

    def refine_if_needed(self, user_id: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        if spec.get("domain","custom") != "custom":
            return spec  # דומיין מוכר - לא נוגעים
        entities = spec.get("entities") or []
        has_fields = any(e.get("fields") for e in entities)
        if entities and has_fields and spec.get("core_behavior"):
            return spec  # כבר מפורט מספיק

        schema = (
            '{"entities":[{"name":"string","fields":[["string","int|float|str|bool"]] }],'
            '"core_behavior":{"name":"string","inputs":["string"],"output":"string",'
            '"formula":"string (weighted sum or rule set)","weights":[1..5 of number],'
            '"tests":[{"inputs":{"string":number},"expected":number}]},'
            '"notes":"string"}'
        )
        prompt = (
            "Given a plain-language app idea, propose domain entities (3-8) with typed fields, "
            "and define ONE core behavior with a numeric output that captures the domain’s essence. "
            "Return JSON ONLY. Keep inputs small and testable. Idea:\n"
            + (spec.get("__source_text__") or "custom domain")
        )
        res = self.gw.structured(user_id=user_id, task="spec_refine", intent="build_app",
                                 schema_hint=schema, prompt=prompt, temperature=0.2)
        j = res["json"]
        spec.setdefault("entities", j.get("entities", []))
        spec["core_behavior"] = j.get("core_behavior", {})
        spec.setdefault("__notes__", j.get("notes",""))
        return spec