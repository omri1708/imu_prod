# imu_repo/ui/schema_extract.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple

def _as_dict(obj: Any) -> Any:
    if obj is None: return None
    if isinstance(obj, (str, int, float, bool)): return obj
    if isinstance(obj, dict): return {k:_as_dict(v) for k,v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_as_dict(x) for x in obj]
    if hasattr(obj, "to_dict"): return _as_dict(obj.to_dict())
    if hasattr(obj, "__dict__"): 
        return {k:_as_dict(v) for k,v in obj.__dict__.items() if not k.startswith("_")}
    return obj

# מאפיינים שדרכם נזהה bind למקור נתונים
_BIND_KEYS = ("endpoint","data_url","ws_url","source")

def extract_table_specs(page_obj: Any) -> List[Dict[str,Any]]:
    """
    שולף מה־DSL של ה־UI את כל טבלאות הנתונים + ציפיות הסכימה עבורן.
    מצופה שב־props של טבלה יהיו:
      - endpoint/data_url/ws_url/source (קישור למקור)
      - columns: [{id/name, type (string|number|bool|date|datetime), unit?, required?}, ...]
      - filters/sort (לשימוש עתידי; כאן בודקים קיום עמודות ותמיכה טיפוסית בסיסית)
    פלט: [{path, binding_url, columns, filters, sort}]
    """
    page = _as_dict(page_obj)
    out: List[Dict[str,Any]] = []

    def walk(node: Any, path: str):
        if isinstance(node, dict):
            kind = node.get("kind") or node.get("type")
            if kind == "table":
                props = node.get("props", {})
                bind_url = ""
                for k in _BIND_KEYS:
                    if isinstance(props.get(k), str):
                        bind_url = props[k]; break
                cols_raw = props.get("columns", [])
                columns: List[Dict[str,Any]] = []
                for c in cols_raw or []:
                    if not isinstance(c, dict): 
                        continue
                    name = c.get("id") or c.get("name")
                    if not name: 
                        continue
                    columns.append({
                        "name": str(name),
                        "type": (c.get("type") or "string").lower(),
                        "unit": c.get("unit"),
                        "required": bool(c.get("required", False))
                    })
                spec = {
                    "path": path or "page",
                    "binding_url": bind_url,
                    "columns": columns,
                    "filters": _as_dict(props.get("filters")),
                    "sort": _as_dict(props.get("sort"))
                }
                out.append(spec)
            # המשך סריקה לכל ילד
            for k,v in node.items():
                p = f"{path}.{k}" if path else k
                walk(v, p)
        elif isinstance(node, list):
            for i, itm in enumerate(node):
                walk(itm, f"{path}[{i}]")
    walk(page, "")
    return out