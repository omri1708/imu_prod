# server/synth_adapter_api.py
# API: create synthesized adapter, list generated kinds, reload registry.
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List
from pathlib import Path
import json

from adapters.synth.generator import SynthSpec, create_adapter
from adapters.synth.registry import reload_registry, list_kinds as list_registered, find_contract

router = APIRouter(prefix="/synth/adapter", tags=["synth-adapter"])

class ParamSpec(BaseModel):
    type: str = "string"
    required: bool = False
    pattern: str | None = None
    enum: List[str] | None = None
    minLength: int | None = None
    default: Any | None = None

class CreateReq(BaseModel):
    name: str
    kind: str
    version: str = "1.0.0"
    description: str = ""
    params: Dict[str, ParamSpec]
    os_templates: Dict[str, str]  # linux/mac/win/any
    examples: Dict[str, Any] = {}
    capabilities: List[str] = []

@router.post("/create")
def create(req: CreateReq):
    # basic guard: kind slug
    if len(req.kind.strip())<3: raise HTTPException(400,"bad kind")
    # generate
    spec = SynthSpec(
        name=req.name, kind=req.kind, version=req.version,
        description=req.description, 
        params={k:v.dict() for k,v in req.params.items()},
        os_templates=req.os_templates, examples=req.examples, 
        capabilities=req.capabilities
    )
    meta = create_adapter(spec)
    reload_registry()
    return {"ok": True, "meta": meta}

@router.get("/list")
def list_adapters():
    return {"ok": True, "kinds": list_registered()}

@router.post("/reload")
def reload():
    out = reload_registry()
    return {"ok": True, "count": len(out)}

@router.get("/contract")
def contract(kind: str):
    p = find_contract(kind)
    if not p: 
        raise HTTPException(404, "contract not found")
    return json.loads(Path(p).read_text(encoding="utf-8"))