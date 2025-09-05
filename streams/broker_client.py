# streams/broker_client.py
from __future__ import annotations
import asyncio, json, os
from typing import Dict, Any, Optional
from adapters.contracts.base import ResourceRequired, record_event

class WSClient:
    def __init__(self, url: str):
        self.url = url
        self._ws = None

    async def _ensure(self):
        try:
            import websockets  # third-party
        except Exception:
            raise ResourceRequired("python-websockets",
                "pip install websockets==12.*",
                "Required to publish realtime events to WS broker")
        if self._ws is None:
            self._ws = await websockets.connect(self.url, max_size=8*1024*1024, compression="deflate")

    async def publish(self, topic: str, payload: Dict[str, Any]):
        await self._ensure()
        msg = json.dumps({"topic": topic, "payload": payload}, ensure_ascii=False)
        await self._ws.send(msg)

    async def close(self):
        if self._ws:
            await self._ws.close()
            self._ws = None

# Helper sync wrapper
def publish_sync(url: str, topic: str, payload: Dict[str, Any]):
    async def _run():
        cli = WSClient(url); await cli.publish(topic, payload); await cli.close()
    try:
        asyncio.run(_run())
    except ResourceRequired as r:
        # bubble up cleanly; upper layer decides
        raise
    except Exception as e:
        record_event("ws_publish_error", {"topic": topic, "err": str(e)})