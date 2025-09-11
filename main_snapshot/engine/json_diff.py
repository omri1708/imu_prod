# imu_repo/engine/json_diff.py
from __future__ import annotations
from typing import Any, List

_SENTINEL = object()

def _is_prim(x: Any) -> bool:
    return isinstance(x, (str, int, float, bool)) or x is None

def _path_join(p: str, key: str) -> str:
    return f"{p}.{key}" if p else key

def diff_paths(a: Any, b: Any, *, _p: str = "") -> List[str]:
    """
    מחזיר רשימת מסלולים (dot-paths) ששונו בין a ל-b.
    בהשוואה:
      - פרימיטיביים: השוואה ישירה.
      - dict: איחוד מפתחות והשוואה רקורסיבית.
      - list/tuple: לפי אינדקס עד min(lenA,lenB) + הוספה/מחיקה.
      - טיפוסים אחרים: השוואה לפי str(x).
    """
    out: List[str] = []
    # שתי פרימיטיביים
    if _is_prim(a) and _is_prim(b):
        if a != b:
            out.append(_p or "$")
        return out
    # dict
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a.keys()) | set(b.keys())
        for k in sorted(keys):
            av = a.get(k, _SENTINEL)
            bv = b.get(k, _SENTINEL)
            if av is _SENTINEL or bv is _SENTINEL:
                out.append(_path_join(_p, str(k)) or "$")
            else:
                out.extend(diff_paths(av, bv, _p=_path_join(_p, str(k))))
        return out
    # list/tuple
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        m = min(len(a), len(b))
        for i in range(m):
            out.extend(diff_paths(a[i], b[i], _p=_path_join(_p, f"[{i}]")))
        if len(a) != len(b):
            out.append(_path_join(_p, f"[{m}:{max(len(a),len(b))}]"))
        return out
    # טיפוסים שונים — השוואה טקסטואלית
    if str(a) != str(b):
        out.append(_p or "$")
    return out