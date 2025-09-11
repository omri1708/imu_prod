# imu_repo/ui/introspect.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple

def _as_dict(obj: Any) -> Any:
    # תומך ב־dataclass/אובייקט/דאקט/ליסט
    if obj is None: return None
    if isinstance(obj, (str, int, float, bool)): return obj
    if isinstance(obj, dict): return {k: _as_dict(v) for k,v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_as_dict(x) for x in obj]
    # נסה להמיר אובייקט עם __dict__ או to_dict
    if hasattr(obj, "to_dict"): return _as_dict(obj.to_dict())
    if hasattr(obj, "__dict__"): 
        return {k:_as_dict(v) for k,v in obj.__dict__.items() if not k.startswith("_")}
    return obj  # last resort

_BIND_KEYS = {"endpoint", "source", "data_url", "ws_url", "rpc", "bind", "expr"}

def extract_ui_claims(page_obj: Any) -> List[Dict[str,Any]]:
    """
    סורק את עץ ה־UI ומחזיר רשימת claims על מקורות נתונים/בינדים.
    כל claim כולל: kind, path, source_url (אם ידוע), meta.
    """
    page = _as_dict(page_obj)
    claims: List[Dict[str,Any]] = []

    def walk(node: Any, path: str):
        if isinstance(node, dict):
            # חפש קישורים/בינדים
            for k,v in node.items():
                p = f"{path}.{k}" if path else k
                if k in _BIND_KEYS and isinstance(v, str):
                    claims.append({
                        "kind": "ui:binding",
                        "path": p,
                        "source_url": v,
                        "meta": {"key": k}
                    })
                # דפדף פנימה
                walk(v, p)
        elif isinstance(node, list):
            for i, itm in enumerate(node):
                walk(itm, f"{path}[{i}]")
        else:
            return

    walk(page, "")
    # הסר כפילויות (אותו path+url)
    seen: set[Tuple[str,str]] = set()
    out: List[Dict[str,Any]] = []
    for c in claims:
        sig = (c["path"], c.get("source_url",""))
        if sig in seen: 
            continue
        seen.add(sig)
        out.append(c)
    return out