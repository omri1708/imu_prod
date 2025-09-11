#!/usr/bin/env python3
import os, sys, json, time, urllib.request, urllib.error, asyncio
import websockets

API_BASE = os.getenv("API_BASE")  # e.g. http://127.0.0.1:8000
WS_URL   = os.getenv("WS_URL")    # e.g. ws://127.0.0.1:8766/ws/wfq?topic=timeline

def post_event(note: str):
    data = json.dumps({
        "topic": "timeline",
        "producer": "ws-ci",
        "priority": 5,
        "event": {"type": "event", "note": note}
    }).encode("utf-8")
    req = urllib.request.Request(API_BASE + "/events/publish",
                                 data=data,
                                 headers={"Content-Type":"application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        if r.status != 200:
            raise RuntimeError(f"publish failed: {r.status}")

async def run():
    assert API_BASE and WS_URL, "API_BASE and WS_URL must be set"
    received = []

    async def ws_task():
        nonlocal received
        async with websockets.connect(WS_URL) as ws:
            t0 = time.time()
            while time.time() - t0 < 8:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    o = json.loads(msg)
                    note = str(o.get("note",""))
                    if o.get("producer") == "ws-ci" and note.startswith("ws-ci-echo-"):
                        received.append(note)
                        if len(received) >= 3:
                            return
                except asyncio.TimeoutError:
                    pass

    async def pub_task():
        # תן ל-WS להתחבר
        await asyncio.sleep(0.5)
        for i in range(3):
            post_event(f"ws-ci-echo-{i}")
            await asyncio.sleep(0.3)

    await asyncio.gather(ws_task(), pub_task())
    if len(received) < 3:
        raise SystemExit("did not receive 3 echoes via WS")

if __name__ == "__main__":
    asyncio.run(run())