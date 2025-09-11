# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, List, Type, Optional, get_type_hints
from sqlmodel import SQLModel, Field

# Base class for all dynamic entities
class BaseEntity(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

_TYPE = {"int": int, "float": float, "str": str, "bool": bool}


def _sanitize(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_]", "_", name or "Entity")


def _field_tuple_list(fields: Any) -> List[List[str]]:
    # normalize: [[name, type], ...]
    out: List[List[str]] = []
    for f in (fields or []):
        if isinstance(f, (list, tuple)) and len(f) >= 2:
            out.append([str(f[0]), str(f[1])])
    return out


def build_models_from_entities_spec(entities: List[Dict[str, Any]]) -> Dict[str, Type[BaseEntity]]:
    reg: Dict[str, Type[BaseEntity]] = {}
    for e in entities or []:
        name = _sanitize(e.get("name") or "Entity")
        fields = list(e.get("fields") or [])
        annotations = {"id": Optional[int]}
        ns = {"__tablename__": name.lower(), "id": Field(default=None, primary_key=True)}

        for f in fields:
            fname, ftype = (f[0], f[1]) if isinstance(f, (list, tuple)) and len(f) >= 2 else (None, None)
            if not fname or fname == "id":
                continue
            ns[fname] = Field(default=None)
            annotations[fname] = _TYPE.get(ftype, str)
        ns["__annotations__"] = annotations
        reg[name] = type(name, (BaseEntity,), ns)

        ns["__annotations__"] = annotations
        cls = type(name, (BaseEntity,), ns)
        reg[name] = cls
    if not reg:
        # Fallback: simple 'Item' entity
        class Item(BaseEntity, table=True):
            name: Optional[str] = Field(default=None)
            value: Optional[float] = Field(default=0.0)
        reg["Item"] = Item
    return reg




        
  





           

