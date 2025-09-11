# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from engine.llm_gateway import LLMGateway

router = APIRouter(prefix="/llm", tags=["llm"])
GW = LLMGateway()

@router.post("/chat")
def chat(body: Dict[str, Any]):
    uid = (body.get("user_id") or "user").strip()
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
         raise HTTPException(400, "prompt required")
    grounded = bool(body.get("grounded"))
    out = GW.chat(user_id=uid, task="chat", intent="free",
                  content={"prompt": prompt, "sources": body.get("sources") or [], "context": body.get("context") or {}},
                  require_grounding=grounded, temperature=float(body.get("temperature", 0.3)))
    return out
