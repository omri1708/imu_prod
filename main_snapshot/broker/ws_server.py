# broker/ws_server.py
import asyncio, json, time
import websockets
from websockets.server import WebSocketServerProtocol
from typing import Dict, Set, DefaultDict
from collections import defaultdict

class StreamBroker:
    def __init__(self):
        self.subs: DefaultDict[str, Set[WebSocketServerProtocol]] = defaultdict(set)

    async def serve(self, host="0.0.0.0", port=8765):
        async def handler(ws: WebSocketServerProtocol):
            topic = None
            try:
                # topic from querystring? websockets lib exposes path
                path = ws.path  # "/stream?topic=xyz"
                if "topic=" in path:
                    topic = path.split("topic=",1)[1]
                if topic:
                    self.subs[topic].add(ws)
                while True:
                    msg = await ws.recv()
                    # client commands are optional; broker broadcasts only server messages
                    # For now, ignore client->server.
            except Exception:
                pass
            finally:
                if topic and ws in self.subs[topic]:
                    self.subs[topic].remove(ws)

        async with websockets.serve(handler, host, port, max_queue=32, ping_interval=20):
            await asyncio.Future()  # run forever

    async def publish(self, topic: str, typ: str, payload: dict):
        ev = {"topic": topic, "type": typ, "ts": int(time.time()*1000), "payload": payload}
        dead = []
        for ws in list(self.subs[topic]):
            try:
                await ws.send(json.dumps(ev))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.subs[topic].discard(ws)

BROKER = StreamBroker()