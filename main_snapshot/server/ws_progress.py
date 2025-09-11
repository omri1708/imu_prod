# server/ws_progress.py
from __future__ import annotations
import asyncio, json, time
from websockets.server import serve

SUBSCRIBERS = set()

async def _broadcast_dict(d: dict):
    if not SUBSCRIBERS:
        return
    msg = json.dumps(d, ensure_ascii=False)
    await asyncio.gather(*[w.send(msg) for w in list(SUBSCRIBERS)], return_exceptions=True)

async def handler(ws):
    SUBSCRIBERS.add(ws)
    try:
        # הודעת חיבור
        await _broadcast_dict({"type":"join","ts":time.time(),"peer":str(id(ws))})
        async for raw in ws:
            try:
                o = json.loads(raw)
                # אם יצרן דוחף "progress"/"event" — נשדר לכל
                if o.get("type") in {"progress","event","timeline"}:
                    o.setdefault("ts", time.time())
                    await _broadcast_dict(o)
                else:
                    # ping/ack או הודעה לא מוכרת
                    await ws.send(json.dumps({"type":"ack","ts":time.time()}))
            except Exception:
                await ws.send(json.dumps({"type":"ack","ts":time.time()}))
    finally:
        SUBSCRIBERS.discard(ws)
        await _broadcast_dict({"type":"leave","ts":time.time(),"peer":str(id(ws))})

async def start_ws(host="0.0.0.0", port=8765):
    async with serve(handler, host, port, ping_interval=20, ping_timeout=20):
        print(f"[WS] listening on ws://{host}:{port}")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(start_ws())