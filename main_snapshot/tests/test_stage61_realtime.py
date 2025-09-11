# imu_repo/tests/test_stage61_realtime.py
from __future__ import annotations
import asyncio, time, socket, base64, os
from typing import Tuple
from realtime.ws_server import WSServer
from realtime.ws_proto import _b, _u, OP_TEXT, OP_PING, OP_CLOSE, send_close, send_ping, recv_frame
from engine.runtime_bridge import apply_runtime_gates

HOST="127.0.0.1"; PORT=8976

async def client_roundtrip(n: int=50, rate_hz: float=200.0) -> float:
    """
    לקוח WS מינימלי: handshake ידני, שליחת טקסטים, מדידת RTT ממחרוזת eco.
    מחזיר p95 שנמדד בצד שרת (נגיש דרך gate בשלב מאוחר יותר).
    """
    r,w = await asyncio.open_connection(HOST, PORT)
    key = base64.b64encode(os.urandom(16)).decode()
    req = (
        f"GET /chat HTTP/1.1\r\n"
        f"Host: {HOST}:{PORT}\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        f"Sec-WebSocket-Version: 13\r\n"
        f"Origin: http://example.com\r\n"
        f"\r\n"
    )
    w.write(_b(req)); await w.drain()
    await r.readuntil(b"\r\n\r\n")  # response headers

    # פונקציות מינימליות ללקוח: שליחת פריימים (client חייב למסך)
    async def send_text(text: str):
        data = text.encode("utf-8")
        b1 = 0x80 | 0x1  # FIN+TEXT
        ln = len(data)
        mask = os.urandom(4)
        if ln < 126:
            header = bytes([b1, 0x80 | ln]) + mask
        elif ln<=0xFFFF:
            header = bytes([b1, 0x80 | 126]) + (len(data)).to_bytes(2,"big") + mask
        else:
            header = bytes([b1, 0x80 | 127]) + (len(data)).to_bytes(8,"big") + mask
        masked = bytes(b ^ mask[i%4] for i,b in enumerate(data))
        w.write(header+masked); await w.drain()

    # שלח n הודעות בקצב נתון
    period = 1.0/max(1.0, rate_hz)
    for i in range(n):
        await send_text(f"hello-{i}")
        await asyncio.sleep(period)

    # קרא n תשובות (echo עם eco-id)
    got=0
    while got<n:
        op, fin, payload = await recv_frame(r)
        if op==OP_TEXT:
            got+=1
        elif op==OP_CLOSE:
            break

    await send_close(w)
    try: 
        w.close(); await w.wait_closed()
    except Exception: ...
    return 0.0

async def run_test():
    # שרת עם handler echo
    async def handler(s: str) -> str:
        # סימולציית עיבוד קצרה
        await asyncio.sleep(0.002)
        return f"echo:{s}"

    srv = WSServer(HOST, PORT, max_queue=64, allowed_origins=["http://example.com"], handler=handler)
    await srv.start()

    # שלח עומס בינוני
    await client_roundtrip(n=80, rate_hz=300.0)

    # Gate: p95 RTT חובה < 120ms ו-backlog < 64
    extras = {"streaming":{"p95_rtt_ms_max": 120.0, "max_queue_depth": 64}}
    gates_out = apply_runtime_gates(extras, stream_metrics=srv.metrics)
    assert gates_out["streaming"]["ok"], f"streaming gate failed: {gates_out['streaming']}"

    # שלח עומס גדול (לחריגה)
    await client_roundtrip(n=400, rate_hz=800.0)
    extras2 = {"streaming":{"p95_rtt_ms_max": 15.0, "max_queue_depth": 8}}
    try:
        apply_runtime_gates(extras2, stream_metrics=srv.metrics)
        raise SystemExit(1)  # היה צריך לזרוק
    except RuntimeError as e:
        # מצופה: streaming_gate_failed
        pass

    await srv.stop()
    print("OK")
    return 0

if __name__=="__main__":
    asyncio.run(run_test())