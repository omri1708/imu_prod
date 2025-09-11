# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, List
from server.dialog.memory_bridge import MB
from fastapi import APIRouter
import os, glob, json, shutil

router = APIRouter(prefix="/chat/memory", tags=["memory"])

USER_ROOT = "./assurance_store_users"

@router.post("/reset")
def reset_memory(body: Dict[str,Any]):
    uid = (body.get("user_id") or "user").strip()
    level = (body.get("level") or "t0").lower()   # t0|t1|t2|all
    # t0 – הקשר קצר (קיים כבר ב-/chat/reset, אבל נשמור אחידות)
    if level == "t0":
        from server.dialog.memory_bridge import MB
        MB.reset_t0(uid)
        return {"ok": True, "cleared": ["t0"]}
    # t1/t2 – קבצי פרופיל/פרסונה
    cleared = []
    if level in ("t1","all"):
        for p in glob.glob(os.path.join(USER_ROOT, f"{uid}*.json")):
            try:
                 os.remove(p)
                 cleared.append(p)
            except Exception:
             pass
    if level in ("t2","all"):
        # אם יש מבני משנה לפרסונה, מחק גם
        sub = os.path.join(USER_ROOT, uid, "persona")
        if os.path.isdir(sub):
            shutil.rmtree(sub, ignore_errors=True)
            cleared.append(sub)
    return {"ok": True, "cleared": cleared, "level": level}

def remember_turn(uid: str, role: str, text: str) -> None:
    MB.observe_turn(uid, role, text)

def history(uid: str, limit: int = 40) -> List[Dict[str, Any]]:
    ctx = MB.pack_context(uid, "")
    return (ctx.get("t0_recent") or [])[-limit:]
