# imu_repo/tests/test_stage61b_realtime_deflate.py
from __future__ import annotations
import asyncio, os, base64, time, zlib
from typing import Tuple
from realtime.ws_server import WSServer
from realtime.ws_proto import recv_frame, send_close, OP_TEXT, OP_BIN, OP_CLOSE

HOST="127.0.0.1"; PORT=8991

async def ws_client_offer_pmd_and_send(payload: bytes, *, binary: bool, fragment: bool) -> list[bytes]:
    r,w = await asyncio.open_connection(HOST, PORT)
    key = base64.b64encode(os.urandom(16)).decode()
    req = (
        f"GET /rt HTTP/1.1\r\n"
        f"Host: {HOST}:{PORT}\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        f"Sec-WebSocket-Version: 13\r\n"
        f"Sec-WebSocket-Extensions: permessage-deflate; client_no_context_takeover\r\n"
        f"Origin: http://example.com\r\n"
        f"\r\n"
    )
    w.write(req.encode()); await w.drain()
    await r.readuntil(b"\r\n\r\n")

    async def send_ws_frame(op:int, data:bytes, fin:bool=True, mask:bool=True):
        b1 = (0x80 if fin else 0x00) | (op & 0x0F)
        ln=len(data)
        if ln<126:
            hdr = bytes([b1, (0x80 if mask else 0x00) | ln])
        elif ln<=0xFFFF:
            hdr = bytes([b1, (0x80 if mask else 0x00) | 126]) + ln.to_bytes(2,"big")
        else:
            hdr = bytes([b1, (0x80 if mask else 0x00) | 127]) + ln.to_bytes(8,"big")
        m = os.urandom(4) if mask else b""
        body = bytes(b ^ m[i%4] for i,b in enumerate(data)) if mask else data
        w.write(hdr + m + body); await w.drain()

    # שליחה מפורקת או לא
    if not fragment:
        await send_ws_frame(OP_BIN if binary else OP_TEXT, payload, fin=True)
    else:
        CH=10_000
        head = payload[:CH]
        await send_ws_frame(OP_BIN if binary else OP_TEXT, head, fin=False)
        i=CH
        while i < len(payload):
            nxt = payload[i:i+CH]; i+=CH
            fin=(i>=len(payload))
            await send_ws_frame(0x0, nxt, fin=fin)  # CONT

    # קבלה של תשובות עד סגירה יזומה שלנו
    outs=[]
    t0=time.time()
    while time.time()-t0 < 1.0:
        try:
            op, fin, rsv1, rsv2, rsv3, pl = await asyncio.wait_for(recv_frame(r), timeout=0.2)
        except asyncio.TimeoutError:
            break
        if op in (OP_TEXT, OP_BIN):
            outs.append(pl)
        elif op==OP_CLOSE:
            break

    await send_close(w)
    try:
        w.close(); await w.wait_closed()
    except Exception: ...
    return outs

async def run_test():
    async def handler(x):
        if isinstance(x, bytes):
            # בזינארי — נהפוך/נשכפל למבחן
            return x[::-1] + x[:4]
        else:
            return x.upper()

    srv = WSServer(HOST, PORT, max_queue=64, allowed_origins=["http://example.com"], handler=handler, chunk_size=8_192)
    await srv.start()

    # 1) טקסט ארוך (פרגמנטציה + דיפלייט ביציאה)
    txt = ("hello-"*5000).encode()
    outs = await ws_client_offer_pmd_and_send(txt, binary=False, fragment=True)
    assert any(b"|ECHO:" in o.upper() or b"|HELLO-" in o.upper() for o in outs)

    # 2) בינארי ארוך (פרגמנטציה); השרת יחזיר bytes[::-1] + 4 הראשונים
    binp = os.urandom(120_000)
    outs2 = await ws_client_offer_pmd_and_send(binp, binary=True, fragment=True)
    assert len(outs2)>=1
    # בדיקת נכונות לוגית על פלט דחוס/לא — נחפש את 4 הבתים הראשונים בסוף
    sig = binp[:4]
    assert any(o.endswith(sig) for o in outs2), "binary echo pattern missing"

    await srv.stop()
    print("OK")
    return 0

if __name__=="__main__":
    asyncio.run(run_test())