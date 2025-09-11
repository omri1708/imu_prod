# imu_repo/grounded/value_checks.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime
from grounded.type_system import canon, is_compatible

class RuntimeRowError(Exception): ...
class TypeViolation(RuntimeRowError): ...
class MissingRequired(RuntimeRowError): ...
class FilterMismatch(RuntimeRowError): ...
class SortMismatch(RuntimeRowError): ...

def _parse_date(s: str) -> datetime:
    # תומך ב־YYYY-MM-DD ו־ISO8601 בסיסי
    try:
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            return datetime.strptime(s, "%Y-%m-%d")
        return datetime.fromisoformat(s.replace("Z","+00:00"))
    except Exception:
        raise TypeViolation(f"invalid date/datetime literal: {s!r}")

def _coerce(v: Any, t: str) -> Any:
    t = canon(t)
    if v is None:
        return None
    if t == "string":
        return str(v)
    if t == "number":
        if isinstance(v, (int, float)): return v
        try: return float(v)
        except Exception: raise TypeViolation(f"expected number, got {type(v).__name__}: {v!r}")
    if t == "bool":
        if isinstance(v, bool): return v
        if isinstance(v, str): 
            ls = v.lower()
            if ls in ("true","1","yes"): return True
            if ls in ("false","0","no"): return False
        raise TypeViolation(f"expected bool, got {type(v).__name__}: {v!r}")
    if t in ("date","datetime"):
        if isinstance(v, (int,float)):  # timestamp
            return datetime.fromtimestamp(float(v))
        if isinstance(v, str):
            return _parse_date(v)
        raise TypeViolation(f"expected date/datetime, got {type(v).__name__}: {v!r}")
    return v

def check_required_and_types(row: Dict[str,Any], columns: List[Dict[str,Any]]) -> None:
    for c in columns or []:
        name = c["name"]
        if c.get("required", False) and (name not in row or row.get(name) in (None,"")):
            raise MissingRequired(f"missing required column '{name}'")
        if name in row:
            _ = _coerce(row[name], c.get("type","string"))

def _pass_filter(value: Any, flt: Dict[str,Any]) -> bool:
    op = str(flt.get("op") or flt.get("operator") or "==").lower()
    target = flt.get("value")
    if op in ("==","="):
        return value == target
    if op in ("!=","<>"):
        return value != target
    if op in (">",">=","<","<="):
        try:
            a = float(value); b = float(target)
        except Exception:
            return False
        if op == ">":  return a >  b
        if op == ">=": return a >= b
        if op == "<":  return a <  b
        if op == "<=": return a <= b
    if op == "in":
        try: 
            return value in list(target)  # type: ignore
        except Exception:
            return False
    if op == "contains":
        try: 
            return str(target) in str(value)
        except Exception:
            return False
    if op == "prefix":
        try: 
            return str(value).startswith(str(target))
        except Exception:
            return False
    if op == "suffix":
        try:
            return str(value).endswith(str(target))
        except Exception:
            return False
    return True  # לא מוכר — לא נחסום (DX)

def check_filters(row: Dict[str,Any], columns: List[Dict[str,Any]], filters: Optional[Dict[str,Any]]) -> None:
    if not filters: 
        return
    for name, rule in (filters or {}).items():
        if name not in row:
            raise FilterMismatch(f"filter column '{name}' missing in row")
        if isinstance(rule, dict):
            if not _pass_filter(row[name], rule):
                raise FilterMismatch(f"row fails filter on '{name}': {rule}")
        else:
            # פורמט לא מוכר — לא נחסום (DX)
            continue

def check_sort(rows: List[Dict[str,Any]], sort_spec: Optional[Dict[str,Any]]) -> None:
    if not sort_spec or not rows:
        return
    col = sort_spec.get("by") or sort_spec.get("column")
    direction = str(sort_spec.get("dir","asc")).lower()
    if not col or col not in rows[0]:
        return  # לא נאכוף אם אין עמודה
    key_vals = [rows[i].get(col) for i in range(len(rows))]
    # נבדוק שהרשימה כבר ממוינת; השוואה שלא תשבור סוגים שונים
    def _cmp_seq(seq: List[Any], reverse: bool) -> bool:
        try:
            sorted_seq = sorted(seq, reverse=reverse)
            return seq == sorted_seq
        except Exception:
            # אם לא ניתן למיין (סוגים שונים) — אל תחסום (DX)
            return True
    rev = (direction == "desc")
    if not _cmp_seq(key_vals, reverse=rev):
        raise SortMismatch(f"rows are not sorted by {col} {direction}")