# server/stream_wfq_ws.py
# WS גשר ל־WFQ Broker: לקוחות נרשמים ל־topic ומקבלים זרם אירועים הוגן/מדורג.
# הרצה: python3 server/stream_wfq_ws.py

from __future__ import annotations
import asyncio, json, time, urllib.parse
from websockets.server import serve, WebSocketServerProtocol
from typing import Dict, Any
from .stream_wfq import BROKER

HELLO = {"type":"hello","msg":"wfq-broker-online"}

async def _producer(topic: str, ws: WebSocketServerProtocol):
    """סשן שידור ללקוח יחיד: פולינג לא חוסם מתוך ה-WFQ והעברה לקליינט."""
    await ws.send(json.dumps(HELLO))
    while True:
        await asyncio.sleep(0.02)  # 50Hz פול קטנטן
        batch = BROKER.poll(topic, max_items=100)
        if not batch:
            continue
        for ev in batch:
            try:
                await ws.send(json.dumps(ev, ensure_ascii=False))
            except Exception:
                return

async def ws_handler(ws: WebSocketServerProtocol):
    # נושא מתוך path: /ws/wfq?topic=timeline
    try:
        q = urllib.parse.urlparse(ws.path).query
        params = urllib.parse.parse_qs(q)
        topic = (params.get("topic") or ["timeline"])[0]
        # אם הנושא טרם קונפג, ניצור ערוץ ברירות מחדל
        BROKER.ensure_topic(topic, rate=50.0, burst=200, weight=2 if topic=="timeline" else 1)
        await _producer(topic, ws)
    except Exception:
        try: await ws.close()
        except Exception: pass

async def main(host="0.0.0.0", port=8766):
    async with serve(ws_handler, host, port, ping_interval=20, ping_timeout=20):
        print(f"[WFQ-WS] ws://{host}:{port}/ws/wfq?topic=<topic>")
        await asyncio.Future()

if __name__=="__main__":
    asyncio.run(main())