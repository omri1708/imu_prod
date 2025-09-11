# server/ws.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set, Any
import asyncio
from collections import defaultdict

router = APIRouter()
_topics: Dict[str, Set[WebSocket]] = defaultdict(set)
_lock = asyncio.Lock()

async def _safe_send(ws: WebSocket, data: Any):
    try:
        await ws.send_json(data)
    except Exception:
        pass

@router.websocket("/topic/{name}")
async def subscribe(ws: WebSocket, name: str):
    await ws.accept()
    async with _lock:
        _topics[name].add(ws)
    try:
        while True:
            await ws.receive_text()  # keepalive / client pings
    except WebSocketDisconnect:
        async with _lock:
            _topics[name].discard(ws)

def push_progress(topic: str, event: dict):
    # Callable מסנכרון: מפזר הודעה לכל המנויים
    async def _broadcast():
        async with _lock:
            clients = list(_topics.get(topic, []))
        # עדיפות: לא לחסום — שולחים במקביל
        await asyncio.gather(*[_safe_send(c, event) for c in clients], return_exceptions=True)

    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.create_task(_broadcast())
    else:
        loop.run_until_complete(_broadcast())