# imu_repo/grounded/schema_consistency.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
from grounded.type_system import is_compatible, canon
from provenance.provenance import evidence_expired, aggregate_trust

class SchemaError(Exception): ...
class SchemaMissing(SchemaError): ...
class ColumnMissing(SchemaError): ...
class TypeMismatch(SchemaError): ...
class UnitMismatch(SchemaError): ...
class NotEnoughSchemaSources(SchemaError): ...
class LowSchemaTrust(SchemaError): ...

def _match_schema_evidences(evs: List[Dict[str,Any]], url: str) -> List[Dict[str,Any]]:
    # עדיפות ל-kind="schema", התאמה לפי source_url (או prefix)
    candidates = []
    for e in evs:
        if e.get("kind") not in ("schema","docs","openapi","db_schema"): 
            continue
        src = str(e.get("source_url",""))
        if not src: 
            continue
        if src == url or url.startswith(src.rstrip("/")) or src.startswith(url.rstrip("/")):
            if not evidence_expired(e):
                candidates.append(e)
    return candidates

def _collect_schema_columns(e: Dict[str,Any]) -> Dict[str,Dict[str,Any]]:
    """
    מצפה בתוכן הראיה (payload) אחד מהפורמטים:
      - {"columns":[{"name","type","unit"?}, ...]}
      - {"schema":{"columns":[...]}}
      - {"components":{"schemas":{...}}}  (OpenAPI — יקח flat מאפיין "type" ו"format")
    מחזיר dict name -> {"type":..,"unit":..}
    """
    p = e.get("payload", {}) or {}
    cols = []
    if isinstance(p.get("columns"), list):
        cols = p["columns"]
    elif isinstance(p.get("schema"), dict) and isinstance(p["schema"].get("columns"), list):
        cols = p["schema"]["columns"]
    elif isinstance(p.get("components"), dict) and isinstance(p["components"].get("schemas"), dict):
        # OpenAPI very-lite: flatten first object-like schema (best-effort)
        for _, sch in p["components"]["schemas"].items():
            props = (sch.get("properties") or {})
            for name, meta in props.items():
                t = meta.get("type") or meta.get("format") or "string"
                cols.append({"name":name, "type":t})
            break
    out: Dict[str,Dict[str,Any]] = {}
    for c in cols:
        name = c.get("name") or c.get("id")
        if not name: 
            continue
        out[str(name)] = {"type": canon(c.get("type","string")), "unit": c.get("unit")}
    return out

def _merge_schemas(schema_list: List[Dict[str,Dict[str,Any]]]) -> Dict[str,Dict[str,Any]]:
    """
    איחוד נאיבי: אם יש התנגשות טיפוסים — נשמור את הראשון; הבדיקה תרד בהמשך לפי התאמה.
    (ניתן להקשיח לרוב־קולות בעתיד)
    """
    merged: Dict[str,Dict[str,Any]] = {}
    for sch in schema_list:
        for name, meta in sch.items():
            if name not in merged:
                merged[name] = dict(meta)
    return merged

def check_table_schema(
    table_spec: Dict[str,Any],
    evidences: List[Dict[str,Any]],
    *,
    min_schema_sources: int,
    min_schema_trust: float
) -> Dict[str,Any]:
    url = table_spec.get("binding_url") or ""
    if not url:
        # טבלה ללא binding — אין מה לאמת
        return {"ok": True, "checked": 0, "sources": 0, "agg_trust": 1.0}
    schemas = _match_schema_evidences(evidences, url)
    if not schemas:
        raise SchemaMissing(f"no schema evidences for {url}")
    agg = aggregate_trust(schemas)
    if len(schemas) < int(min_schema_sources):
        raise NotEnoughSchemaSources(f"need >= {min_schema_sources} schema sources, got {len(schemas)}")
    if agg < float(min_schema_trust):
        raise LowSchemaTrust(f"agg_schema_trust {agg:.2f} < {min_schema_trust:.2f}")

    colmaps = [_collect_schema_columns(e) for e in schemas]
    merged = _merge_schemas(colmaps)
    checked = 0
    for col in table_spec.get("columns") or []:
        name = col["name"]
        if name not in merged:
            if col.get("required", False):
                raise ColumnMissing(f"required column '{name}' missing in schema")
            else:
                # אם לא required — אפשר לאפשר המשך (DX); נבדוק התאמות לאחר fetch בזמן ריצה
                continue
        want_t = col.get("type","string")
        got_t  = merged[name].get("type","string")
        if not is_compatible(want_t, got_t):
            raise TypeMismatch(f"column '{name}' type {want_t} !~ {got_t}")
        want_u = col.get("unit")
        got_u  = merged[name].get("unit")
        if want_u and got_u and str(want_u) != str(got_u):
            raise UnitMismatch(f"column '{name}' unit {want_u} != {got_u}")
        checked += 1

    return {"ok": True, "checked": checked, "sources": len(schemas), "agg_trust": agg}