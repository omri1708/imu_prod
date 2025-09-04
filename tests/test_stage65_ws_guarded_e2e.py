# imu_repo/tests/test_stage65_ws_guarded_e2e.py
from __future__ import annotations
import asyncio, os, base64, json
from typing import Any
from realtime.ws_proto import recv_frame, OP_TEXT
from realtime.ws_guarded_server import WSGuardedServer
from grounded.claims import current
from security.response_signer import verify_payload

HOST="127.0.0.1"; PORT=8766

async def _client_once(msg: str) -> dict:
    r,w = await asyncio.open_connection(HOST, PORT)
    key = base64.b64encode(os.urandom(16)).decode()
    # בקשת השידכום
    req = (
        f"GET /rt HTTP/1.1\r\n"
        f"Host: {HOST}:{PORT}\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        f"Sec-WebSocket-Version: 13\r\n"
        f"Origin: http://example.com\r\n"
        f"\r\n"
    )
    w.write(req.encode()); await w.drain()
    await r.readuntil(b"\r\n\r\n")

    # שלח טקסט קצר (ללא פרגמנטציה)
    def _mask(b: bytes) -> bytes:
        m = os.urandom(4)
        return bytes([b[i]^m[i%4] for i in range(len(b))]), m
    data = msg.encode("utf-8")
    b1 = 0x80 | 0x1  # FIN|TEXT
    ln = len(data)
    if ln<126:
        hdr = bytes([b1, 0x80 | ln])
    elif ln<=0xFFFF:
        hdr = bytes([b1, 0x80 | 126]) + ln.to_bytes(2,"big")
    else:
        hdr = bytes([b1, 0x80 | 127]) + ln.to_bytes(8,"big")
    masked, mask = _mask(data)
    w.write(hdr + mask + masked); await w.drain()

    # קבל תשובה אחת
    op, fin, rsv1, rsv2, rsv3, pl = await recv_frame(r)
    assert op==OP_TEXT
    # הורד eco-id אם קיים
    s = pl.decode("utf-8","replace")
    p = s.find("|")
    if p>0: s = s[p+1:]
    obj = json.loads(s)
    try:
        w.close(); await w.wait_closed()
    except Exception: ...
    return obj

async def run_test():
    # בנה שרת Guarded עם handler שמוסיף ראיות אמיתיות לפי הקלט
    async def handler(x: Any) -> str:
        # טוען ראיה — כאן מדגים תוכן+meta
        cur = current()
        cur.add_evidence(f"proof-for:{x}", {"source_url":"https://example.test", "trust":0.92, "ttl_s":30})
        return f"ok:{x}"

    srv = WSGuardedServer(HOST, PORT, allowed_origins=["http://example.com"], min_trust=0.7)
    await srv.set_handler(handler)
    await srv.start()

    obj = await _client_once("ping")
    assert obj.get("text")=="ok:ping"
    claims = obj.get("claims") or []
    assert len(claims)==1 and "digest" in claims[0]
    # חתימה
    assert verify_payload(obj) is True

    await srv.stop()
    print("OK")
    return 0

if __name__=="__main__":
    asyncio.run(run_test())