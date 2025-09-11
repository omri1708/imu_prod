# common/ws_progress.py
from __future__ import annotations
import asyncio, json, time
from typing import Dict, Any, Optional

try:
    import websockets # pip install websockets
except Exception:
    websockets=None

class WSProgress:
    def __init__(self, url:str, topic:str):
        self.url=url; self.topic=topic
    async def emit(self, kind:str, data:Dict[str,Any]):
        if not websockets:
            print("[WS disabled]", kind, data); return
        msg={"topic": self.topic, "ts": time.time(), "kind": kind, "data": data}
        async with websockets.connect(self.url, max_size=1<<24) as ws:
            await ws.send(json.dumps(msg))