# imu_repo/ui/schema_compose.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple

_BIND_KEYS = ("endpoint","data_url","ws_url","source")

def _walk_to_path(root: Any, path: str) -> Tuple[Any, Any]:
    """מחזיר (parent, key) לצומת המצוין ב-path בסגנון a.b[2].c"""
    if not path:
        return None, None
    cur = root; parent = None; key = None
    i = 0; token = ""
    tokens: List[Any] = []
    while i < len(path):
        ch = path[i]
        if ch == ".":
            if token: tokens.append(token); token = ""
        elif ch == "[":
            if token: tokens.append(token); token = ""
            j = path.find("]", i)
            tokens.append(int(path[i+1:j]))
            i = j
        else:
            token += ch
        i += 1
    if token: tokens.append(token)

    for t in tokens:
        parent, key = cur, t
        if isinstance(t, int) and isinstance(cur, list):
            cur = cur[t]
        elif isinstance(t, str) and isinstance(cur, dict):
            cur = cur.get(t)
        else:
            return None, None
    return parent, key

def _merge_columns(dst_cols: List[Dict[str,Any]], src_cols: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    """מיזוג לפי name/id; לא מוחקים שדות קיימים בלתי מוכרים."""
    index = { (c.get("id") or c.get("name")): i for i,c in enumerate(dst_cols or []) }
    out = list(dst_cols or [])
    for sc in (src_cols or []):
        sid = sc.get("name") or sc.get("id")
        if not sid:
            continue
        entry = {
            "id": sid,
            "name": sid,
            "type": sc.get("type","string"),
            "unit": sc.get("unit"),
            "required": bool(sc.get("required", False)),
        }
        if sid in index:
            out[index[sid]].update({k:v for k,v in entry.items() if v is not None})
        else:
            out.append(entry)
    return out

def apply_table_spec_patch(page_obj: Any, spec: Dict[str,Any], *, mode: str = "merge") -> bool:
    """משקף columns/filters/sort ו-binding_url ל-node בטבלה. mode: 'merge' או 'overwrite'."""
    parent, key = _walk_to_path(page_obj, spec.get("path",""))
    if parent is None:
        return False
    node = parent[key]
    if not isinstance(node, dict):
        return False
    props = node.setdefault("props", {})

    # binding_url -> אחד ממפתחות ה-bind
    bind_url = spec.get("binding_url")
    if bind_url:
        for k in _BIND_KEYS:
            props[k] = bind_url
            break

    # columns
    if "columns" in spec:
        if mode == "overwrite":
            props["columns"] = [
                {"id": c.get("name"), "name": c.get("name"), "type": c.get("type","string"),
                 "unit": c.get("unit"), "required": bool(c.get("required", False))}
                for c in (spec["columns"] or [])
            ]
        else:  # merge
            props["columns"] = _merge_columns(props.get("columns") or [], spec["columns"] or [])

    # filters / sort
    if "filters" in spec:
        props["filters"] = spec["filters"]
    if "sort" in spec:
        props["sort"] = spec["sort"]
    return True

def apply_table_specs(page_obj: Any, specs: List[Dict[str,Any]], *, mode: str = "merge") -> int:
    n = 0
    for s in (specs or []):
        if apply_table_spec_patch(page_obj, s, mode=mode):
            n += 1
    return n
