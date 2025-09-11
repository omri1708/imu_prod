# imu_repo/tests/test_stage105_realtime_strict.py
from __future__ import annotations
import asyncio, json, struct, socket
from typing import Dict, Any, Tuple
import pytest

from realtime.tcp_framed import TCPFramedServer
from realtime.strict_sink import StrictSink, Reject
from realtime.protocol import pack, unpack

async def _client_send(sock: socket.socket, op: str, bundle: Dict[str, Any]) -> Dict[str, Any]:
    msg = pack(op, bundle)
    hdr = struct.pack(">I", len(msg))
    sock.sendall(hdr + msg)
    # recv
    rcv_hdr = await asyncio.get_running_loop().sock_recv(sock, 4)
    (n,) = struct.unpack(">I", rcv_hdr)
    data = await asyncio.get_running_loop().sock_recv(sock, n)
    _, out_bundle = unpack(data)
    return out_bundle

@pytest.mark.asyncio
async def test_realtime_strict_sink_no_leakage():
    # policy מחמירה: דורש מקור אחד לפחות וניקוד אמון >=1
    policy = {"min_distinct_sources": 1, "min_total_trust": 1.0,
              "perf_sla": {"latency_ms": {"p95_max": 200.0}}}

    async def handler(op: str, bundle: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        # מנסה להחזיר את אותו bundle – StrictSink יפיל אם אין claims
        return op, bundle

    sink = StrictSink(policy)
    srv = TCPFramedServer("127.0.0.1", 0, handler, sink)
    await srv.start()
    port = srv._server.sockets[0].getsockname()[1]  # type: ignore

    loop = asyncio.get_running_loop()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(False)
    await loop.sock_connect(sock, ("127.0.0.1", port))

    # hello
    hdr = await loop.sock_recv(sock, 4)
    (n,) = struct.unpack(">I", hdr)
    hello = await loop.sock_recv(sock, n)
    op, bundle = unpack(hello)
    assert op == "control/hello" and bundle.get("ok") is True

    # 1) ניסיון לשלוח bundle בלי claims -> אמור לקבל control/error
    msg = pack("app/echo", {"text": "hi-no-claims"})
    sock.sendall(struct.pack(">I", len(msg)) + msg)
    rcv_hdr = await loop.sock_recv(sock, 4)
    (n,) = struct.unpack(">I", rcv_hdr)
    data = await loop.sock_recv(sock, n)
    op, bundle = unpack(data)
    assert op == "control/error"
    assert bundle.get("reason") in ("bundle_missing_fields", "bad_types_or_empty_claims")

    # 2) שולחים bundle עם claims + evidence
    good = {
        "text": "sum 2+2 = 4",
        "claims": [{
            "type": "compute",
            "text": "2+2=4",
            "evidence": [{"kind": "compute", "expr": "2+2", "value": 4, "source": "local"}]
        }]
    }
    resp = await _client_send(sock, "app/echo", good)
    assert resp["text"].startswith("sum 2+2")
    assert isinstance(resp.get("_verifier_meta"), dict)
    assert resp["_verifier_meta"]["min_distinct_sources"] >= 1

    sock.close()