# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, List, Type, Optional, get_type_hints
from sqlmodel import SQLModel, Field
"""
Dynamic SQLModel builders:
- Given entities spec => build SQLModel classes at runtime.
- Supported field types: int, float, str, bool (extensible).
"""



# Base class for all dynamic entities
class BaseEntity(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

_TYPE_MAP = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
}

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
    registry: Dict[str, Type[BaseEntity]] = {}
    for e in entities or []:
        name = _sanitize(e.get("name") or "Entity")
        fields = _field_tuple_list(e.get("fields"))
        annotations = {"id": Optional[int]}
        namespace = {"__tablename__": name.lower(), "id": Field(default=None, primary_key=True)}

        for fname, ftype in fields:
            if fname == "id":
                continue
            py_t = _TYPE_MAP.get(ftype, str)
            annotations[fname] = py_t
            namespace[fname] = Field(default=None)

        namespace["__annotations__"] = annotations
        cls = type(name, (BaseEntity,), namespace)
        registry[name] = cls
    if not registry:
        # Fallback: simple 'Item' entity
        class Item(BaseEntity, table=True):
            name: Optional[str] = Field(default=None)
            value: Optional[float] = Field(default=0.0)
        registry["Item"] = Item
    return registry
