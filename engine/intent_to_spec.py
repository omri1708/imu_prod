# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from typing import Dict, Any
from engine.llm_gateway import LLMGateway
from engine.domain_kb import DOMAIN_FITNESS

class IntentToSpec:
    """
    ממיר טקסט חופשי לבקשה מובנית (SPEC) עם מושגים דומייניים.
    משתמש ב-LLMGateway במצב JSON מוגבל + מאמת מול הידע הדומייני.
    """
    def __init__(self):
        self.gateway = LLMGateway()

    def from_text(self, user_id: str, text: str) -> Dict[str, Any]:
        schema = (
            '{"domain":"fitness_training|custom",'
            '"personas":[{"name":"string","goals":["string"]}],'
            '"features":["string"],'
            '"data_inputs":["string"],'
            '"non_functionals":["string"],'
            '"platforms":["web|mobile|desktop|api"],'
            '"integrations":["string"],'
            '"constraints":["string"],'
            '"success_metrics":["string"],'
            '"examples":{"workout":"string","plan":"string"}}'
        )
        res = self.gateway.structured(
            user_id=user_id, task="intent", intent="build_app",
            schema_hint=schema,
            prompt=f"נתח ובנה SPEC לאפליקציה מן הטקסט הבא (עברית/English). החזר JSON בלבד.\n{text}",
            temperature=0.0
        )
        spec = res["json"]

        # אימות בסיסי + הזרקת ידע דומייני אם domain=fitness_training
        domain = spec.get("domain","custom")
        if domain == "fitness_training":
            spec.setdefault("entities", DOMAIN_FITNESS["entities"])
            spec.setdefault("events",   DOMAIN_FITNESS["events"])
            spec.setdefault("rules",    DOMAIN_FITNESS["rules"])
            # ודא שיש תכונות דומייניות לא ריקות
            if not spec.get("features"):
                spec["features"] = ["personalized_plan", "progress_tracking", "session_logging", "recommendations"]
            if not spec.get("success_metrics"):
                spec["success_metrics"] = ["plan_adherence_rate","weekly_active_minutes","injury_risk_reduction_score"]
        else:
            # custom: עדיין נחזיר מבנה, אבל בלי להעמיד פנים
            spec.setdefault("entities", [])
            spec.setdefault("events", [])
            spec.setdefault("rules", [])
        spec["__provenance__"] = {"llm_gateway": True}
        return spec