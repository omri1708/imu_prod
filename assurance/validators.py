# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, Callable, List, Tuple

class Validator:
    def __init__(self, name: str, fn: Callable[[Dict[str, Any]], Tuple[bool, str]]):
        self.name = name
        self.fn = fn

    def run(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        ok, msg = self.fn(context)
        return ok, f"{self.name}: {msg}"

def schema_validator(schema: Dict[str, Any]) -> Validator:
    """
    Minimal schema validator (types/ranges/required).
    Schema example:
      {
        "required": ["title", "value"],
        "properties": {
          "title": {"type":"string", "minLength":1},
          "value": {"type":"number", "minimum": 0, "maximum": 100}
        }
      }
    """
    def _check(ctx: Dict[str, Any]) -> Tuple[bool, str]:
        props = schema.get("properties", {})
        data  = ctx.get("artifact", {})
        # required
        for k in schema.get("required", []):
            if k not in data:
                return False, f"missing required '{k}'"
        # properties
        for k, rules in props.items():
            if k not in data: 
                continue
            v = data[k]
            t = rules.get("type")
            if t:
                if t == "string" and not isinstance(v, str): return False, f"{k} not string"
                if t == "number" and not isinstance(v, (int, float)): return False, f"{k} not number"
                if t == "integer" and not isinstance(v, int): return False, f"{k} not integer"
                if t == "boolean" and not isinstance(v, bool): return False, f"{k} not boolean"
            if isinstance(v, str) and rules.get("minLength") is not None and len(v) < rules["minLength"]:
                return False, f"{k} minLength {rules['minLength']}"
            if isinstance(v, (int,float)):
                if rules.get("minimum") is not None and v < rules["minimum"]:
                    return False, f"{k} < minimum"
                if rules.get("maximum") is not None and v > rules["maximum"]:
                    return False, f"{k} > maximum"
        return True, "ok"
    return Validator("schema", _check)

def unit_range_validator(field: str, unit: str, minimum: float | None = None, maximum: float | None = None) -> Validator:
    def _check(ctx: Dict[str, Any]) -> Tuple[bool, str]:
        data = ctx.get("artifact", {})
        if field not in data:
            return False, f"{field} missing"
        v = data[field]
        if not isinstance(v, (int,float)):
            return False, f"{field} not number"
        if minimum is not None and v < minimum: return False, f"{field} < {minimum}{unit}"
        if maximum is not None and v > maximum: return False, f"{field} > {maximum}{unit}"
        return True, "ok"
    return Validator(f"unit_range[{field}{unit}]", _check)
