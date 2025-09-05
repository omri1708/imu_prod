# imu_repo/tests/test_stage106_ui_stream.py
from __future__ import annotations
import asyncio, struct, socket
import pytest
from typing import Dict, Any, Tuple
from realtime.tcp_framed import TCPFramedServer
from realtime.strict_sink import StrictSink
from realtime.protocol import pack, unpack
from ui.dsl_runtime_rt import UISession, TableWidget

@pytest.mark.asyncio
async def test_ui_session_blocks_unjustified():
    policy = {"min_distinct_sources": 1, "min_total_trust": 1.0, "perf_sla": {"latency_ms":{"p95_max":200}}}
    async def handler(op: str, bundle: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        return op, bundle

    sink = StrictSink(policy)
    srv = TCPFramedServer("127.0.0.1", 0, handler, sink)
    await srv.start()
    port = srv._server.sockets[0].getsockname()[1]  # type: ignore

    loop = asyncio.get_running_loop()
    sock = socket.socket(); sock.setblocking(False)
    await loop.sock_connect(sock, ("127.0.0.1", port))

    # hello
    hdr = await loop.sock_recv(sock, 4)
    (n,) = struct.unpack(">I", hdr)
    await loop.sock_recv(sock, n)

    # שולחים הודעה בלי claims => השרת יחסום ויחזיר control/error
    msg = pack("ui/update", {"text":"bad","ui":{"orders_table":{"rows":[{"id":1,"sku":"A","qty":1}]}}})
    sock.sendall(struct.pack(">I", len(msg))+msg)
    hdr = await loop.sock_recv(sock, 4)
    (n,) = struct.unpack(">I", hdr)
    data = await loop.sock_recv(sock, n)
    op, bundle = unpack(data)
    assert op == "control/error"

    # הודעה תקפה עם claims+evidence => עוברת, וצד הקליינט יעדכן טבלה
    good = {
        "text": "orders update",
        "claims": [{"type":"compute","text":"ok","evidence":[{"kind":"internal","source":"local"}]}],
        "ui": {"orders_table":{"rows":[{"id":2,"sku":"B","qty":3}]}}
    }
    msg = pack("ui/update", good)
    sock.sendall(struct.pack(">I", len(msg))+msg)
    hdr = await loop.sock_recv(sock, 4)
    (n,) = struct.unpack(">I", hdr)
    data = await loop.sock_recv(sock, n)
    op, bundle = unpack(data)
    assert op == "ui/update"

    # קליינט מקומי: UISession יאמת שוב ויעדכן
    ui = UISession(min_sources=1, min_trust=1.0)
    tbl = TableWidget(key_field="id")
    ui.register("orders_table", tbl)
    ui.handle_stream_message({"op":"ui/update","bundle":good})
    rows = tbl.to_list()
    assert any(r["id"]==2 and r["sku"]=="B" for r in rows)
    sock.close()