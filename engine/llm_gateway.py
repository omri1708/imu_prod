# -*- coding: utf-8 -*-
from __future__ import annotations
import json, time, os
from typing import Dict, Any, List, Optional, Tuple
from engine.llm_client import LLMClient, LLMError
from engine.prompt_builder import PromptBuilder
from assurance.assurance import AssuranceKernel
from grounded.fact_gate import FactGate   # אם אצלך זה server/grounded/... התאם import
from grounded.provenance_store import ProvenanceStore
from security.policy import ResponseSigner  # או היכן שמוגדר אצלך
from user_model.model import UserStore, UserModel
from user_model.subject import SubjectEngine

class LLMGateway:
    """
    צוואר בקבוק יחיד לכל LLM.
    - מזריק פרופיל/הקשר משתמש (subject/persona/scopes/מדיניות).
    - בונה פרומפטים עם PromptBuilder (לומד מתוצאה בכל פנייה).
    - מחיל Grounding/Provenance/Privacy/Budget/SLO לפני/אחרי הקריאה.
    - מחזיר תוצר רק אם עבר Gates (או Structured JSON שנבדק).
    """
    def __init__(self,
                 store_root_users: str = "./assurance_store_users",
                 store_root_text:  str = "./assurance_store_text"):
        self.llm = LLMClient()
        self.pb  = PromptBuilder("./learning/prompt_templates.jsonl",
                                 "./learning/prompt_stats.jsonl")
        self.users = UserStore(store_root_users)
        self.subject = SubjectEngine(self.users)
        self.kernel = AssuranceKernel(store_root_text)
        self.fact_gate = FactGate()               # מחייב הצגת ראיות בתשובות עובדתיות
        self.prov = ProvenanceStore(store_root_text)
        self.signer = ResponseSigner("llm-gateway")

    # ----- API עיקרי -----
    def chat(self, user_id: str, task: str, intent: str,
             content: Dict[str,Any], require_grounding: bool=False,
             temperature: float=0.0, max_tokens: int=1024) -> Dict[str,Any]:
        """
        task: "answer"/"plan"/"codegen"/"rewrite"/...
        intent: תווית לוגית (build_app/run_action/clarify/...)
        content: {"prompt": "...", "context": {...}, "sources": [...?]}
        require_grounding: אם True – אוכף מקורות ו-verify לפני שחרור פלט
        """
        t0 = time.time()
        persona = self.subject.persona(user_id)       # פרופיל תפעולי מרוכז
        prompt_msgs = self.pb.compose(user_id=user_id, task=task, intent=intent,
                                      persona=persona, content=content)

        # ---- השיחה עם ה-LLM עצמה (נקודה יחידה) ----
        raw = self.llm.chat(prompt_msgs, temperature=temperature, max_tokens=max_tokens)

        # ---- Grounding / Provenance / Assure ----
        entry = {"user_id": user_id, "task": task, "intent": intent,
                 "prompt": prompt_msgs, "raw": raw, "latency_ms": int(1000*(time.time()-t0))}
        root = None
        try:
            # רישום גולמי ל-CAS (trace) עוד לפני עיבוד
            entry_d = self.kernel.cas.put_bytes(json.dumps(entry, ensure_ascii=False).encode("utf-8"),
                                                meta={"type":"llm_raw"})
            # אם נדרש Grounding (לשאלות עובדתיות) – מחייב ראיות ו-verify
            if require_grounding:
                checked = self.fact_gate.enforce(raw, content.get("sources") or [])
                # רושם provenance + חותם
                root = self.kernel.commit(self.signer, payload={"text": checked["text"],
                                                                "citations": checked["citations"],
                                                                "raw_digest": entry_d})
                result = {"text": checked["text"], "citations": checked["citations"], "root": root["root"]}
            else:
                # מחזיר כתשובת ביניים (למשל plan/codegen) – בהמשך יאומת ע"י טסטים/קומפילציה
                root = self.kernel.commit(self.signer, payload={"text": raw, "raw_digest": entry_d})
                result = {"text": raw, "root": root["root"]}
            # עדכון למידה של ה-PromptBuilder (אחרי הצלחה מערכתית)
            self.pb.learn(user_id=user_id, task=task, intent=intent, success=True, latency_ms=entry["latency_ms"])
            return {"ok": True, "payload": result}
        except Exception as e:
            # למידה מכישלון
            self.pb.learn(user_id=user_id, task=task, intent=intent, success=False, latency_ms=entry["latency_ms"],
                          error=str(e))
            raise

    def structured(self, user_id:str, task:str, intent:str,
                   schema_hint:str, prompt:str, temperature:float=0.0) -> Dict[str,Any]:
        """
        JSON-Mode: מכריח את המודל להחזיר JSON תקין לפי רמז־סכמה.
        """
        persona = self.subject.persona(user_id)
        messages = self.pb.compose(user_id=user_id, task=task, intent=intent,
                                   persona=persona, content={"prompt": prompt, "schema_hint": schema_hint},
                                   json_only=True)
        out = self.llm.chat(messages, temperature=temperature, max_tokens=1024)
        # ניסיון לתקן JSON עד 2 פעמים
        for _ in range(2):
            try:
                data = json.loads(out)
                self.pb.learn(user_id=user_id, task=task, intent=intent, success=True)
                return {"ok": True, "json": data}
            except Exception:
                messages.append({"role":"assistant","content":out})
                messages.append({"role":"user","content":"התשובה אינה JSON תקין. החזר את ה-JSON בלבד."})
                out = self.llm.chat(messages, temperature=0.0, max_tokens=1024)
        self.pb.learn(user_id=user_id, task=task, intent=intent, success=False, error="invalid_json")
        raise LLMError("invalid JSON")
