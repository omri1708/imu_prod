# -*- coding: utf-8 -*-
import os, asyncio, logging
try:
    import websockets  # minimal dependency in container only
except Exception:
    websockets = None

HOST = "0.0.0.0"
PORT = int(os.getenv("WS_PORT", "8766"))

_clients = set()

async def _handler(ws, path):
    _clients.add(ws)
    try:
        async for msg in ws:
            # minimal broadcast (placeholder for WFQ)
            await asyncio.gather(*(c.send(msg) for c in _clients if c is not ws), return_exceptions=True)
    finally:
        _clients.discard(ws)

def main():
    logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL","INFO").upper(), logging.INFO))
    if websockets is None:
        raise RuntimeError("websockets not installed")
    start_server = websockets.serve(_handler, HOST, PORT)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_server)
    loop.run_forever()

if __name__ == "__main__":
    main()
