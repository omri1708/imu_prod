# contracts/schema.py
# -*- coding: utf-8 -*-
import re
from typing import Any, Dict, List
from contracts.errors import ContractViolation

def _type_name(x: Any) -> str:
    if x is None: return "null"
    if isinstance(x, bool): return "boolean"
    if isinstance(x, int) and not isinstance(x, bool): return "integer"
    if isinstance(x, float): return "number"
    if isinstance(x, str): return "string"
    if isinstance(x, list): return "array"
    if isinstance(x, dict): return "object"
    return type(x).__name__

def _expect(cond: bool, msg: str, where: str):
    if not cond:
        raise ContractViolation(f"schema:{msg}", detail={"where": where})

def validate_schema(data: Any, schema: Dict[str, Any], where: str = "$") -> None:
    """תמיכה בסיסית ב-JSON Schema: type/required/properties/items/enum/min/max/len/pattern"""
    st = schema.get("type")
    if st:
        tname = _type_name(data)
        if isinstance(st, list):
            _expect(any(tname == s for s in st), f"type expected {st} got {tname}", where)
        else:
            _expect(tname == st, f"type expected {st} got {tname}", where)

    if "enum" in schema:
        _expect(data in schema["enum"], f"value {data} not in enum", where)

    if st == "number" or st == "integer":
        if "minimum" in schema: _expect(data >= schema["minimum"], f"{data} < minimum", where)
        if "maximum" in schema: _expect(data <= schema["maximum"], f"{data} > maximum", where)

    if st == "string":
        if "minLength" in schema: _expect(len(data) >= schema["minLength"], "string shorter than minLength", where)
        if "maxLength" in schema: _expect(len(data) <= schema["maxLength"], "string longer than maxLength", where)
        if "pattern" in schema: _expect(re.search(schema["pattern"], data) is not None, "pattern not matched", where)

    if st == "array":
        items = schema.get("items")
        if items:
            for i,el in enumerate(data):
                validate_schema(el, items, f"{where}[{i}]")

    if st == "object":
        req = schema.get("required", [])
        props = schema.get("properties", {})
        for k in req:
            _expect(k in data, f"missing required key '{k}'", where)
        for k,v in data.items():
            if k in props:
                validate_schema(v, props[k], f"{where}.{k}")