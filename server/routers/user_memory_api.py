# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, List
from fastapi import APIRouter
import os, glob, json, shutil
from server.dialog.memory_bridge import MB, _load, _save, _now_ms, _hash

router = APIRouter(prefix="/chat/memory", tags=["memory"])

USER_ROOT = "./assurance_store_users"


@router.post("/reset")
def reset_memory(body: Dict[str,Any]):
    uid   = (body.get("user_id") or "user").strip()
    level = (body.get("level") or "t0").lower()   # "t0" | "deep"
    if level == "t0":
        MB.reset_t0(uid); return {"ok": True, "cleared": ["t0"]}
    MB.wipe_user(uid); return {"ok": True, "cleared": ["t0","t1","t2"]}

@router.post("/put")
def put_memory(body: Dict[str,Any]):
    """הוספת רשומה ל-T2 (למשל עובדה/העדפה) עם scope/ttl/consent."""
    uid = (body.get("user_id") or "user").strip()
    fact = str(body.get("fact","")).strip()
    scope = str(body.get("scope","t2")).lower()
    ttl_s = int(body.get("ttl_s", 365*24*3600))
    consent = bool(body.get("consent", True))
    if not fact: return {"ok": False, "error":"empty_fact"}
    st = _load(uid)
    rec = {"ts": _now_ms(), "hash": _hash(fact), "fact": fact, "scope": scope, "ttl_s": ttl_s, "consent": consent}
    st.setdefault("t2",[]).append(rec); _save(uid, st)
    return {"ok": True, "added": rec}

@router.post("/forget")
def forget_memory(body: Dict[str,Any]):
    """מחיקה לפי hash או by-substring."""
    uid = (body.get("user_id") or "user").strip()
    key = str(body.get("key","")).strip()
    st = _load(uid)
    before = len(st.get("t2",[]))
    if key:
        st["t2"] = [r for r in st.get("t2",[]) if r.get("hash") != key and key not in (r.get("fact") or "")]
        _save(uid, st)
        return {"ok": True, "removed": before - len(st["t2"])}
    return {"ok": False, "error":"key_required"}

def remember_turn(uid: str, role: str, text: str) -> None:
    MB.observe_turn(uid, role, text)

def history(uid: str, limit: int = 40) -> List[Dict[str, Any]]:
    ctx = MB.pack_context(uid, "")
    return (ctx.get("t0_recent") or [])[-limit:]
