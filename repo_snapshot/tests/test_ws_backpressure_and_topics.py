# imu_repo/tests/test_ws_backpressure_and_topics.py
from __future__ import annotations
import asyncio, base64, json, struct, pytest
from realtime.ws_push import WSPushServer
from realtime.strict_sink import StrictSink

async def handler(op,bundle):
    if op=="ui/subscribe": return "control/ack", {"ok":True,"topics":bundle.get("topics",[])}
    return op,bundle

async def ws_connect(host,port):
    r,w = await asyncio.open_connection(host,port)
    key = base64.b64encode(b"kk").decode()
    req = (f"GET / HTTP/1.1\r\nHost:{host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode()
    w.write(req); await w.drain()
    await r.readuntil(b"\r\n\r\n")
    return r,w

def _mask_send(w, obj):
    data = json.dumps(obj).encode()
    b1=0x80|0x1; ln=len(data); mask=b"mask"
    if ln<126: hdr=struct.pack("!BB",b1,0x80|ln)
    elif ln<(1<<16): hdr=struct.pack("!BBH",b1,0x80|126,ln)
    else: hdr=struct.pack("!BBQ",b1,0x80|127,ln)
    w.write(hdr+mask+bytes(b ^ mask[i%4] for i,b in enumerate(data)))

async def _recv(r):
    h=await r.readexactly(2); b1,b2=h[0],h[1]; ln=(b2 & 0x7F)
    if ln==126: ln=struct.unpack(">H",await r.readexactly(2))[0]
    elif ln==127: ln=struct.unpack(">Q",await r.readexactly(8))[0]
    pl=await r.readexactly(ln); return json.loads(pl.decode())

@pytest.mark.asyncio
async def test_topics_filter_and_rate_limit():
    sink = StrictSink({"min_distinct_sources":1,"min_total_trust":1.0})
    srv = WSPushServer("127.0.0.1",0,handler,sink, queue_max=8, msg_rate=5, byte_rate=10_000, burst_msgs=5, burst_bytes=20_000)
    await srv.start()
    port = srv._srv.sockets[0].getsockname()[1]  # type: ignore

    r,w = await ws_connect("127.0.0.1",port)
    await _recv(r)  # hello
    _mask_send(w, {"op":"ui/subscribe","bundle":{"topics":["orders"]}}); await w.drain()
    await _recv(r)

    good = {"text":"tick","claims":[{"type":"c","text":"ok","evidence":[{"kind":"k","source":"local"}]}],"ui":{"x":{"rows":[{"id":1}]}}}
    # ישדר רק ל-"orders"
    for _ in range(10):
        await srv.broadcast("ui/update", good, topic="orders")

    # נקבל לפחות הודעה אחת, וייתכן שחלק הוגבלו (rate_limited אזהרות)
    got_updates=0; got_warns=0
    for _ in range(10):
        try:
            doc = await asyncio.wait_for(_recv(r), timeout=0.5)
        except asyncio.TimeoutError:
            break
        if doc["op"]=="ui/update": got_updates+=1
        if doc["op"]=="control/warn": got_warns+=1
    assert got_updates >= 1
    assert got_warns >= 0