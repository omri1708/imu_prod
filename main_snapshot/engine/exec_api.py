# imu_repo/engine/exec_api.py
from __future__ import annotations
from typing import Dict, Any, List
from exec.cells import run_code
from exec.select import choose
from exec.errors import ResourceRequired, ExecError


def exec_best(task: Dict[str,Any], ctx: Dict[str,Any]) -> Dict[str,Any]:
    """
    task: {"code": "...", "lang": optional, "tags": [...], "cell_name": "..."}
    אם lang לא נתון — נבחר על פי tags + זמינות.
    """
    user_id = (ctx or {}).get("user_id","anon")
    hints = (ctx or {}).get("__routing_hints__", {})
    tags = list(task.get("tags") or [])
    # שילוב מצב רגשי/מטרות: למשל אם user רוצה build_any_app → נטה לשפות system/go
    if hints.get("search_depth")=="deep":
        tags = list(set(tags + ["system","concurrency","enterprise"]))
    lang = task.get("lang")
    if not lang:
        cand = choose(tags)
        if not cand:
            raise ResourceRequired("toolchain", "Install at least one of: Python/Node/Go/JDK/.NET/G++/Rust")
        lang = cand[0]
    res = run_code(lang, task["code"], user_id=user_id, cell_name=task.get("cell_name","cell"))
    # פלט אחיד
    return {"lang":lang, **res}