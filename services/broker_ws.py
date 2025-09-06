# services/broker_ws.py
import asyncio, json, time
import websockets
from collections import defaultdict, deque

MAX_QUEUE=1000
CLIENTS=defaultdict(set)
QUEUES=defaultdict(lambda: deque(maxlen=MAX_QUEUE))

PRIORITY={"progress":1,"event":2,"log":3}

async def producer(topic):
    while True:
        if QUEUES[topic]:
            yield QUEUES[topic].popleft()
        else:
            await asyncio.sleep(0.01)

async def handler(ws):
    # פרוטוקול פשוט: {"topic": "...", "kind":"progress|event|log", "data":{...}}
    async for raw in ws:
        try:
            msg=json.loads(raw)
            topic=msg.get("topic","default")
            msg["ts"]=time.time()
            QUEUES[topic].append(msg)
            # push ללקוחות מחוברים
            for cli in set(CLIENTS[topic]):
                try: await cli.send(json.dumps(msg))
                except: pass
        except Exception as e:
            await ws.send(json.dumps({"error": str(e)}))

async def subscribe(ws, path):
    # נתיב = /stream/<topic>  או /ingest
    if path.startswith("/stream/"):
        topic=path.split("/stream/")[1]
        CLIENTS[topic].add(ws)
        try:
            async for _ in ws: pass
        finally:
            CLIENTS[topic].discard(ws)
    else:
        await handler(ws)

if __name__=="__main__":
    import sys
    port=int(sys.argv[1]) if len(sys.argv)>1 else 8765
    print(f"WS broker on :{port}")
    websockets.serve(subscribe, "0.0.0.0", port)
    asyncio.get_event_loop().run_forever()