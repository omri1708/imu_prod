# adapters/validate.py
from __future__ import annotations
import json, os

def _type_ok(v, t):
    if t=="string": return isinstance(v, str)
    if t=="array": return isinstance(v, list)
    if t=="object": return isinstance(v, dict)
    if t=="number": return isinstance(v,(int,float)) and not isinstance(v,bool)
    return True

def validate_params(schema_path: str, params: dict) -> None:
    if not os.path.exists(schema_path):
        raise ValueError(f"schema not found: {schema_path}")
    schema = json.load(open(schema_path,"r",encoding="utf-8"))
    # required
    for k in schema.get("required", []):
        if k not in params:
            raise ValueError(f"missing param: {k}")
    # types
    props = schema.get("properties",{})
    for k,info in props.items():
        if k in params and "type" in info and not _type_ok(params[k], info["type"]):
            raise ValueError(f"type mismatch for '{k}': expected {info['type']}")
    # enums
    for k,info in props.items():
        if k in params and "enum" in info and params[k] not in info["enum"]:
            raise ValueError(f"value for '{k}' not in enum")