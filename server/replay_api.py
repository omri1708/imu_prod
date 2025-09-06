# server/replay_api.py
# Replay from runbook history only (no external execution).
from __future__ import annotations
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from pathlib import Path
import json, time, asyncio

from server.runbook_history import list_history, load_history
from server.stream_wfq import BROKER

router = APIRouter(prefix="/replay", tags=["replay"])

@router.get("/history")
def history_list():
    return {"ok": True, "items": list_history()}

class ReplayReq(BaseModel):
    run_id_or_path: str
    realtime: bool = False
    speed: float = 4.0   # מהר פי 4 (כשלא realtime)

@router.post("/start")
async def start_replay(req: ReplayReq):
    try:
        rec = load_history(req.run_id_or_path)
    except Exception as e:
        raise HTTPException(404, f"history not found: {e}")
    BROKER.ensure_topic("timeline", rate=100.0, burst=500, weight=2)
    # שידור אירועי ההיסטוריה
    evs = rec.get("events", [])
    if not evs: return {"ok": False, "reason": "no events"}
    t0 = rec.get("ts_start", time.time())
    for e in evs:
        if req.realtime:
            now=time.time()
            delay = max(0.0, float(e.get("ts", now))-t0 - (now - t0))
            await asyncio.sleep(delay)
        else:
            await asyncio.sleep(0.1/req.speed)  # playback מהיר
        ev = {"type": e.get("type","event"), "ts": e.get("ts", time.time()), "note": e.get("note")}
        BROKER.submit("timeline","replay",ev, priority=5)
    return {"ok": True, "count": len(evs)}