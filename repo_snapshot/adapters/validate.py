# adapters/validate.py
from __future__ import annotations
import json, re, os

class SchemaError(ValueError): ...

def _load(schema_path: str) -> dict:
    if not os.path.exists(schema_path):
        raise SchemaError(f"schema not found: {schema_path}")
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _type_ok(v, t: str) -> bool:
    return (t=="string"  and isinstance(v, str)) or \
           (t=="number"  and isinstance(v, (int,float)) and not isinstance(v,bool)) or \
           (t=="integer" and isinstance(v, int) and not isinstance(v,bool)) or \
           (t=="boolean" and isinstance(v, bool)) or \
           (t=="object"  and isinstance(v, dict)) or \
           (t=="array"   and isinstance(v, list))

def validate_params(schema_path: str, params: dict) -> None:
    s = _load(schema_path)
    # required
    for k in s.get("required", []):
        if k not in params:
            raise SchemaError(f"missing param: {k}")
    props = s.get("properties", {})
    for k,v in params.items():
        # additionalProperties
        if k not in props and s.get("additionalProperties", True) is False:
            raise SchemaError(f"unexpected param: {k}")
        info = props.get(k)
        if not info:
            continue
        if "type" in info and not _type_ok(v, info["type"]):
            raise SchemaError(f"type mismatch for '{k}', expected {info['type']}")
        if "enum" in info and v not in info["enum"]:
            raise SchemaError(f"value for '{k}' not in enum {info['enum']}")
        if "pattern" in info and isinstance(v, str) and re.fullmatch(info["pattern"], v) is None:
            raise SchemaError(f"param '{k}' fails pattern: {info['pattern']}")
    # array limits
    for k,info in props.items():
        if info.get("type") == "array" and k in params:
            if "maxItems" in info and len(params[k]) > info["maxItems"]:
                raise SchemaError(f"array '{k}' exceeds maxItems={info['maxItems']}")