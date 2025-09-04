# synth/schema_validate.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple

class SchemaError(Exception): ...

def validate(obj: Any, schema: Dict[str,Any]) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    st = schema.get("type")
    if st:
        if st=="object" and not isinstance(obj, dict): errs.append("type_object")
        if st=="array"  and not isinstance(obj, list): errs.append("type_array")
        if st=="string" and not isinstance(obj, str):  errs.append("type_string")
        if st=="number" and not (isinstance(obj, int) or isinstance(obj, float)): errs.append("type_number")
        if st=="integer" and not isinstance(obj, int): errs.append("type_integer")
        if st=="boolean" and not isinstance(obj, bool): errs.append("type_boolean")
    if isinstance(obj, (int,float)):
        if "minimum" in schema and obj < schema["minimum"]: errs.append("too_small")
        if "maximum" in schema and obj > schema["maximum"]: errs.append("too_large")
    if isinstance(obj, dict):
        req = schema.get("required") or []
        for k in req:
            if k not in obj: errs.append(f"missing:{k}")
        props = schema.get("properties") or {}
        for k, sub in props.items():
            if k in obj:
                ok, e = validate(obj[k], sub)
                if not ok: errs += [f"{k}.{ee}" for ee in e]
    return (len(errs)==0, errs)