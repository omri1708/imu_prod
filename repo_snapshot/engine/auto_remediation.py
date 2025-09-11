# imu_repo/engine/auto_remediation.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable
import copy
import re

# סוגי חריגות חוסמות הנתמכות (ממנועי הגארדים)
from engine.runtime_guard import RuntimeBlocked
from engine.kpi_regression import KPIRegressionBlocked

@dataclass
class Diagnosis:
    kind: str                 # "runtime_missing_required" | "runtime_filter" | "runtime_sort" | "runtime_drift" | "kpi_p95" | "kpi_error_rate" | "kpi_schema"
    detail: str               # הסבר חופשי (לוג)
    evidence: Dict[str, Any]  # נתוני עזר, שמות עמודות/ספים/דלתות, hashes...

@dataclass
class Remedy:
    description: str
    safety: str               # "conservative" | "risky" | "forbidden"
    apply: Callable[[Dict[str,Any], List[Dict[str,Any]]], None]          # func(policy:dict, table_specs:list[dict]) -> None

def _parse_runtime_reason(msg: str) -> Optional[Diagnosis]:
    # נחלץ binding_url אם קיים בתבנית [... url=...]
    def _extract_url(s: str) -> Optional[str]:
        m = re.search(r"url=([^\]\s]+)", s)
        return m.group(1) if m else None
    # דוגמאות: "runtime_row: missing required column 'amount'"
    if "missing required column" in msg:
        m = re.search(r"missing required column '([^']+)'", msg)
        col = m.group(1) if m else "UNKNOWN"
        return Diagnosis("runtime_missing_required", msg, {"column": col, "binding_url": _extract_url(msg)})
    if "row fails filter on" in msg:
        m = re.search(r"row fails filter on '([^']+)'", msg)
        col = m.group(1) if m else "UNKNOWN"
        return Diagnosis("runtime_filter", msg, {"column": col, "binding_url": _extract_url(msg)})
    if "runtime_sort" in msg:
        return Diagnosis("runtime_sort", msg, {"binding_url": _extract_url(msg)})
    if "runtime_drift" in msg:
        # runtime_drift: content_hash_changed PREV -> NEW
        m = re.search(r"content_hash_changed\s+([0-9a-f]{64})\s+->\s+([0-9a-f]{64})", msg)
        prev, new = (m.group(1), m.group(2)) if m else (None, None)
        return Diagnosis("runtime_drift", msg, {"prev_hash": prev, "new_hash": new, "binding_url": _extract_url(msg)})
    return None

def _parse_kpi_reason(msg: str) -> List[Diagnosis]:
    diags: List[Diagnosis] = []
    # p95 regression 35.00ms > 20.0ms; error-rate regression 0.0150 > 0.0100; schema-error-rate regression 0.0050 > 0
    if "p95 regression" in msg:
        m = re.search(r"p95 regression ([\-\d\.]+)ms > ([\d\.]+)ms", msg)
        delta, limit = (float(m.group(1)), float(m.group(2))) if m else (None, None)
        diags.append(Diagnosis("kpi_p95", msg, {"delta_ms": delta, "limit_ms": limit}))
    if "error-rate regression" in msg:
        m = re.search(r"error-rate regression ([\-\d\.]+) > ([\d\.]+)", msg)
        delta, limit = (float(m.group(1)), float(m.group(2))) if m else (None, None)
        diags.append(Diagnosis("kpi_error_rate", msg, {"delta": delta, "limit": limit}))
    if "schema-error-rate regression" in msg:
        m = re.search(r"schema-error-rate regression ([\-\d\.]+) > 0", msg)
        delta = float(m.group(1)) if m else None
        diags.append(Diagnosis("kpi_schema", msg, {"delta": delta}))
    return diags

def diagnose(block_exc: Exception) -> List[Diagnosis]:
    """
    ממפה הודעת חסימה לסט דיאגנוזות נורמליות.
    """
    if isinstance(block_exc, RuntimeBlocked):
        d = _parse_runtime_reason(str(block_exc))
        return [d] if d else [Diagnosis("runtime_unknown", str(block_exc), {})]
    if isinstance(block_exc, KPIRegressionBlocked):
        diags = _parse_kpi_reason(str(block_exc))
        return diags if diags else [Diagnosis("kpi_unknown", str(block_exc), {})]
    # לא מוכר — נחזיר אבחנה כללית
    return [Diagnosis("unknown", str(block_exc), {})]

# ---------- מחולל תיקונים בטוחים (מדיניות קובעת מה מותר) ----------

def _find_table_by_path(table_specs: List[Dict[str,Any]], path: str) -> Optional[Dict[str,Any]]:
    for t in table_specs or []:
        if t.get("path") == path:
            return t
    return None

def _find_table_by_binding_url(table_specs, url: str):
    for t in table_specs or []:
        if t.get("binding_url") == url:
            return t
    return None

def _table_id_from_spec(spec: Dict[str,Any]) -> str:   #TODO- זה נותן איתור של טבלה, איך ניתן לדייק יותר?
    """מזהה יציב ללוגיקה/דריפט: binding_url > path > name."""
    return spec.get("binding_url") or spec.get("path") or spec.get("name") or "<unknown>"

def _relax_required_column(column: str, table_spec: Dict[str,Any]) -> bool:
    cols = table_spec.get("columns") or []
    changed = False
    for c in cols:
        if c.get("name") == column and c.get("required", False):
            c["required"] = False
            changed = True
    return changed

def _remove_filter(column: str, table_spec: Dict[str,Any]) -> bool:
    flt = table_spec.get("filters") or {}
    if column in flt:
        del flt[column]
        table_spec["filters"] = flt
        return True
    return False

def _weaken_sort_if_needed(table_spec: Dict[str,Any]) -> bool:
    """
    אם יש sort, נהפוך אותו ללא מחייב (נמחק sort) — פתרון שמרני למניעת חסימה,
    עד שנקבל בסיס נתונים שמספק מיון.
    """
    if table_spec.get("sort"):
        table_spec["sort"] = None
        return True
    return False

def _raise_kpi_threshold(policy: Dict[str,Any], key: str, delta: float, max_raise: float) -> bool:
    """
    מעלה סף עד תקרה מוגדרת במדיניות auto_raise_limits.
    """
    limits = policy.setdefault("auto_raise_limits", {})
    allowed = float(limits.get(key, 0.0))
    if allowed <= 0.0:
        return False
    # נגדיל את הסף אך לא נעבור את allowed
    if key == "p95_ms":
        curr = float(policy.get("max_p95_increase_ms", 50.0))
        inc = min(max(delta, 0.0), allowed)
        policy["max_p95_increase_ms"] = curr + inc
        # צריכה להיות עקיבה מול ה־Gate
        return True
    if key == "error_rate":
        curr = float(policy.get("max_error_rate_increase", 0.01))
        inc = min(max(delta, 0.0), allowed)
        policy["max_error_rate_increase"] = curr + inc
        return True
    return False


def _allow_new_hash_for(policy: Dict[str,Any], table_id: str, new_hash: Optional[str]) -> bool:
    """מעדכן baseline פר-טבלה: runtime_prev_hash_map[table_id] = new_hash"""
    if not new_hash:
        return False
    if not bool(policy.get("allow_update_prev_hash_on_schema_ok", False)):
        return False
    mp = policy.setdefault("runtime_prev_hash_map", {})
    mp[table_id] = new_hash
    return True

def propose_remedies(diags: List[Diagnosis], *, policy: Dict[str,Any], table_specs: List[Dict[str,Any]]) -> List[Remedy]:
    res: List[Remedy] = []
    # קונטקסט: נסמן טבלת יעד (אם יש) — נשתמש במסלול הראשון
    def _resolve_target_spec(d: Diagnosis) -> Optional[Dict[str,Any]]:
        url = (d.evidence or {}).get("binding_url")
        if url:
            t = _find_table_by_binding_url(table_specs, url)
            if t: return t
        # fallback לפי path של הראשון ברשימה (או הראשון כפשוטו)
        if table_specs:
            p = table_specs[0].get("path")
            t = _find_table_by_path(table_specs, p)
            return t or table_specs[0]
        return None

    for d in diags:
        if d.kind == "runtime_missing_required" and policy.get("allow_relax_required_if_missing", False):
            def _ap(policy:Dict[str,Any], ts:List[Dict[str,Any]]):
                t = _resolve_target_spec(d)
                if t: _relax_required_column(d.evidence.get("column",""), t)
            res.append(Remedy(
                description=f"Relax required on '{d.evidence.get('column')}'",
                safety="risky",  # שינוי חוזה UI — מחייב בקרה
                apply=_ap
            ))
        elif d.kind == "runtime_filter" and policy.get("allow_remove_filter_if_blocked", True):
            def _ap(policy:Dict[str,Any], ts:List[Dict[str,Any]]):
                t = _resolve_target_spec(d)
                if t: _remove_filter(d.evidence.get("column",""), t)
            res.append(Remedy(
                description=f"Remove blocking filter on '{d.evidence.get('column')}'",
                safety="conservative",
                apply=_ap
            ))
        elif d.kind == "runtime_sort" and policy.get("allow_weaken_sort", True):
            def _ap(policy:Dict[str,Any], ts:List[Dict[str,Any]]):
                t = _resolve_target_spec(d)
                if t: _weaken_sort_if_needed(t)
            res.append(Remedy(
                description="Drop strict sort requirement (temporary)",
                safety="conservative",
                apply=_ap
            ))
        elif d.kind == "runtime_drift":
            if policy.get("allow_update_prev_hash_on_schema_ok", False):
                def _ap(policy:Dict[str,Any], ts:List[Dict[str,Any]]):
                    # נזהה את הטבלה שעליה רצים הרמדיז (ב-rollback_guard מעבירים [spec] פר-טבלה)
                    t = _resolve_target_spec(d)
                    if t:
                        table_id = _table_id_from_spec(t)
                        _allow_new_hash_for(policy, table_id, d.evidence.get("new_hash"))
                res.append(Remedy(
                    description="Accept new runtime content hash as baseline (schema already ok)",
                    safety="conservative",
                    apply=_ap
                ))
        elif d.kind == "kpi_p95":
            delta = float(d.evidence.get("delta_ms") or 0.0)
            def _ap(policy:Dict[str,Any], ts:List[Dict[str,Any]]):
                _raise_kpi_threshold(policy, "p95_ms", delta, policy.get("auto_raise_limits",{}).get("p95_ms",0.0))
            res.append(Remedy(
                description=f"Raise p95 allowance by ≤{policy.get('auto_raise_limits',{}).get('p95_ms',0.0)}ms",
                safety="conservative",
                apply=_ap
            ))
        elif d.kind == "kpi_error_rate":
            delta = float(d.evidence.get("delta") or 0.0)
            def _ap(policy:Dict[str,Any], ts:List[Dict[str,Any]]):
                _raise_kpi_threshold(policy, "error_rate", delta, policy.get("auto_raise_limits",{}).get("error_rate",0.0))
            res.append(Remedy(
                description=f"Raise error-rate allowance by ≤{policy.get('auto_raise_limits',{}).get('error_rate',0.0)}",
                safety="risky",  # עליה בשיעור שגיאות — לשימוש זהיר
                apply=_ap
            ))
        # others -> לא נוגעים אוטומטית
    return res

def apply_remedies(remedies: List[Remedy], *, policy: Dict[str,Any], table_specs: List[Dict[str,Any]]) -> None:
    for r in remedies:
        r.apply(policy, table_specs)