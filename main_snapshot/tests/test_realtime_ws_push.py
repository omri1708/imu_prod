# imu_repo/tests/test_realtime_ws_push.py
from __future__ import annotations
import asyncio, socket, base64, hashlib, json, struct
import pytest
from typing import Dict, Any
from realtime.ws_push import WSPushServer
from realtime.strict_sink import StrictSink

async def app_handler(op, bundle):  # echo w/ack
    if op == "ui/subscribe":
        return "control/ack", {"ok": True}
    return op, bundle

async def ws_client_connect(host, port):
    reader, writer = await asyncio.open_connection(host, port)
    key = base64.b64encode(b"clientkey").decode()
    req = (
        "GET / HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "\r\n"
    ).encode()
    writer.write(req); await writer.drain()
    await reader.readuntil(b"\r\n\r\n")
    return reader, writer

async def ws_send_json(writer, obj):
    data = json.dumps(obj).encode()
    b1 = 0x80 | 0x1
    ln = len(data)
    # client frames must be masked
    mask_key = b"mask"
    if ln < 126: header = struct.pack("!BB", b1, 0x80 | ln)
    elif ln < (1<<16): header = struct.pack("!BBH", b1, 0x80 | 126, ln)
    else: header = struct.pack("!BBQ", b1, 0x80 | 127, ln)
    writer.write(header + mask_key + bytes(b ^ mask_key[i % 4] for i, b in enumerate(data)))
    await writer.drain()

async def ws_recv_json(reader):
    hdr = await reader.readexactly(2)
    b1, b2 = hdr[0], hdr[1]
    ln = (b2 & 0x7F)
    if ln == 126: ln = struct.unpack(">H", await reader.readexactly(2))[0]
    elif ln == 127: ln = struct.unpack(">Q", await reader.readexactly(8))[0]
    # server frames unmasked
    payload = await reader.readexactly(ln)
    return json.loads(payload.decode())

@pytest.mark.asyncio
async def test_broadcast_to_two_clients():
    policy = {"min_distinct_sources":1, "min_total_trust":1.0, "perf_sla":{"latency_ms":{"p95_max":200}}}
    sink = StrictSink(policy)
    srv = WSPushServer("127.0.0.1", 0, app_handler, sink)
    await srv.start()
    port = srv._srv.sockets[0].getsockname()[1]  # type: ignore

    r1, w1 = await ws_client_connect("127.0.0.1", port)
    r2, w2 = await ws_client_connect("127.0.0.1", port)
    # קרא hello
    await ws_recv_json(r1); await ws_recv_json(r2)

    # subscribe
    await ws_send_json(w1, {"op":"ui/subscribe", "bundle":{"topics":["orders"]}})
    await ws_recv_json(r1)

    await ws_send_json(w2, {"op":"ui/subscribe", "bundle":{"topics":["orders"]}})
    await ws_recv_json(r2)

    # שדר הודעת UI (Grounded-Strict)
    good = {
        "text":"tick",
        "claims":[{"type":"compute","text":"ok","evidence":[{"kind":"internal","source":"local"}]}],
        "ui":{"orders_table":{"rows":[{"id":1,"sku":"A","qty":2}]}}
    }
    await srv.broadcast("ui/update", good)

    # שני הלקוחות מקבלים
    doc1 = await ws_recv_json(r1)
    doc2 = await ws_recv_json(r2)
    assert doc1["op"] == "ui/update" and doc2["op"] == "ui/update"
    assert "orders_table" in doc1["bundle"]["ui"] and "orders_table" in doc2["bundle"]["ui"]