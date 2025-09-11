from __future__ import annotations
from typing import Any, Dict
from fastapi import APIRouter
from pydantic import BaseModel

from engine.llm.cache import default_cache

router = APIRouter(prefix="/cache", tags=["cache"])
CACHE = default_cache()

class CacheKeyIn(BaseModel):
    key: str

class CachePutIn(BaseModel):
    key: str
    model: str
    payload: Dict[str,Any]
    ttl_s: int = 3600

@router.post("/get")
async def get(inp: CacheKeyIn):
    ok, ent = CACHE.get(inp.key)
    return {"ok": ok, "entry": ent.__dict__ if ent else None}

@router.post("/put")
async def put(inp: CachePutIn):
    ent = CACHE.put(inp.key, model=inp.model, payload=inp.payload, ttl_s=inp.ttl_s)
    return {"ok": True, "entry": ent.__dict__}