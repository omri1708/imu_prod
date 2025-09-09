# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, List
from server.dialog.memory_bridge import MB

def remember_turn(uid: str, role: str, text: str) -> None:
    MB.observe_turn(uid, role, text)

def history(uid: str, limit: int = 40) -> List[Dict[str, Any]]:
    ctx = MB.pack_context(uid, "")
    return (ctx.get("t0_recent") or [])[-limit:]
