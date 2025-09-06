# server/ws_progress.py
# WebSocket broker לדחיפת progress/timeline. אין מוקים.
# הפעלה:  python3 server/ws_progress.py
from __future__ import annotations
import asyncio, json, time
from websockets.server import serve

SUBSCRIBERS = set()

async def handler(ws):
    SUBSCRIBERS.add(ws)
    try:
        async for msg in ws:
            # לקוח יכול לשלוח "ping"—נחזיר ack.
            try:
                o = json.loads(msg)
                if o.get("type") == "ping":
                    await ws.send(json.dumps({"type":"ack","ts":time.time()}))
            except Exception:
                await ws.send(json.dumps({"type":"ack","ts":time.time()}))
    finally:
        SUBSCRIBERS.discard(ws)

async def broadcast(event: dict):
    """שגר אירוע לכל המנויים (משתמשים בה מאז קוד חיצוני/דמואים)."""
    if not SUBSCRIBERS: return
    msg = json.dumps(event)
    await asyncio.gather(*[w.send(msg) for w in list(SUBSCRIBERS)], return_exceptions=True)

async def start_ws(host="0.0.0.0", port=8765):
    async with serve(handler, host, port, ping_interval=20, ping_timeout=20):
        print(f"[WS] listening on ws://{host}:{port}")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(start_ws())