# imu_repo/synth/schema_validate.py
from __future__ import annotations
from typing import Any, Dict, Optional

class ClaimSchemaError(Exception): ...

def _ensure_num(x: Any, name: str) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    raise ClaimSchemaError(f"{name} must be number")

def validate_claim_schema(claim: Dict[str,Any]) -> None:
    """
    תומך ב:
      - {"schema":{"type":"number","unit":"ms|s|pct|any","min":..,"max":..,"tolerance":..}, "value": <number>}
      - {"schema":{"type":"string","min_len":...,"max_len":...}, "value": <str>}
      - {"schema":{"type":"enum","choices":[...]}, "value": <str>}
    """
    schema = claim.get("schema")
    if not schema:
        return  # לא חובה
    typ = schema.get("type")
    if typ == "number":
        v = _ensure_num(claim.get("value"), "value")
        u = (schema.get("unit") or "any").lower()
        if "min" in schema:
            if v < float(schema["min"]):
                raise ClaimSchemaError(f"value {v} < min {schema['min']}")
        if "max" in schema:
            if v > float(schema["max"]):
                raise ClaimSchemaError(f"value {v} > max {schema['max']}")
        if u not in ("ms","s","pct","any"):
            raise ClaimSchemaError(f"unknown unit {u}")
        # tolerance לא נבדק כאן (רק בהצלבה)
    elif typ == "string":
        s = claim.get("value")
        if not isinstance(s, str):
            raise ClaimSchemaError("value must be string")
        if "min_len" in schema and len(s) < int(schema["min_len"]):
            raise ClaimSchemaError("string too short")
        if "max_len" in schema and len(s) > int(schema["max_len"]):
            raise ClaimSchemaError("string too long")
    elif typ == "enum":
        s = claim.get("value")
        choices = schema.get("choices") or []
        if s not in choices:
            raise ClaimSchemaError(f"value '{s}' not in choices")
    else:
        raise ClaimSchemaError(f"unsupported schema type: {typ}")

def consistent_numbers(a: float, b: float, tol: float) -> bool:
    if tol < 0:
        tol = 0.0
    # בדיקת |a-b| <= tol*max(|a|,|b|,1)
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) <= (tol * scale)