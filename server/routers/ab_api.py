from __future__ import annotations
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Dict, List
from engine.llm_gateway import LLMGateway
from engine.llm.dual_infer import run_dual

router = APIRouter(prefix="/ab", tags=["ab"])
GW = LLMGateway()

class Inp(BaseModel):
    text: str

@router.post("/dual")
async def dual(inp: Inp):
    msgs=[{"role":"user","content":inp.text}]
    def _A(): return GW._openai_chat(msgs, json_mode=False, temperature=0.2)
    def _B(): return GW._openai_chat(msgs, json_mode=False, temperature=0.7)
    return run_dual(_A,_B)
