# imu_repo/realtime/ws_minimal.py
from __future__ import annotations
import asyncio, base64, hashlib, struct, json
from typing import Optional, Tuple, Dict, Any, Awaitable, Callable
from realtime.protocol import pack, unpack, ProtocolError
from realtime.strict_sink import StrictSink, Reject

Handler = Callable[[str, Dict[str, Any]], Awaitable[Tuple[str, Dict[str, Any]]]]

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

class WSProtocolError(Exception): pass

class WSConn:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.r = reader
        self.w = writer
        self.closed = False

    async def handshake(self) -> None:
        # קורא בקשת HTTP פשוטה עד CRLF CRLF
        req = await self.r.readuntil(b"\r\n\r\n")
        head = req.decode("latin1")
        if "Upgrade: websocket" not in head and "upgrade: websocket" not in head:
            raise WSProtocolError("not a websocket upgrade")
        # חילוץ Sec-WebSocket-Key
        key_line = None
        for line in head.split("\r\n"):
            if line.lower().startswith("sec-websocket-key:"):
                key_line = line.split(":", 1)[1].strip()
                break
        if not key_line:
            raise WSProtocolError("missing Sec-WebSocket-Key")
        accept = base64.b64encode(hashlib.sha1((key_line + GUID).encode("ascii")).digest()).decode("ascii")
        resp = "HTTP/1.1 101 Switching Protocols\r\n" \
               "Upgrade: websocket\r\n" \
               "Connection: Upgrade\r\n" \
               f"Sec-WebSocket-Accept: {accept}\r\n" \
               "\r\n"
        self.w.write(resp.encode("latin1"))
        await self.w.drain()

    async def recv_text(self) -> str:
        # קורא פריים בודד (מסכות מצד הלקוח): תומך רק opcode=1 (טקסט)
        hdr = await self.r.readexactly(2)
        b1, b2 = hdr[0], hdr[1]
        fin = (b1 >> 7) & 1
        opcode = b1 & 0x0F
        mask = (b2 >> 7) & 1
        length = (b2 & 0x7F)
        if opcode == 8:  # CLOSE
            self.closed = True
            return ""
        if opcode not in (1, 2):  # טקסט/בינארי – כאן תומכים רק בטקסט
            raise WSProtocolError(f"unsupported opcode {opcode}")
        if length == 126:
            ext = await self.r.readexactly(2)
            length = struct.unpack(">H", ext)[0]
        elif length == 127:
            ext = await self.r.readexactly(8)
            length = struct.unpack(">Q", ext)[0]
        if mask != 1:
            raise WSProtocolError("client frames must be masked")
        mask_key = await self.r.readexactly(4)
        data = await self.r.readexactly(length)
        # הסרת מסכה
        unmasked = bytes(b ^ mask_key[i % 4] for i, b in enumerate(data))
        if opcode == 2:
            raise WSProtocolError("binary frames not supported in this minimal impl")
        return unmasked.decode("utf-8", errors="strict")

    async def send_text(self, s: str) -> None:
        data = s.encode("utf-8")
        b1 = 0x80 | 0x1  # FIN + text
        n = len(data)
        if n < 126:
            hdr = bytes([b1, n])
        elif n < (1 << 16):
            hdr = bytes([b1, 126]) + struct.pack(">H", n)
        else:
            hdr = bytes([b1, 127]) + struct.pack(">Q", n)
        self.w.write(hdr + data)
        await self.w.drain()

    async def close(self):
        if not self.closed:
            self.w.write(b"\x88\x00")  # close frame
            try:
                await self.w.drain()
            except Exception:
                pass
            self.closed = True
        self.w.close()
        try:
            await self.w.wait_closed()
        except Exception:
            pass

class WSServer:
    def __init__(self, host: str, port: int, handler: Handler, sink: StrictSink):
        self.host = host
        self.port = port
        self.handler = handler
        self.sink = sink
        self._server: Optional[asyncio.AbstractServer] = None

    async def _client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        peer = writer.get_extra_info("peername")
        conn = WSConn(reader, writer)
        try:
            await conn.handshake()
            # hello
            await conn.send_text('{"op":"control/hello","bundle":{"server":"ws_minimal","ok":true}}')
            while True:
                raw = await conn.recv_text()
                if conn.closed:
                    break
                try:
                    op, bundle = unpack(raw.encode("utf-8"))
                except ProtocolError as e:
                    await conn.send_text('{"op":"control/error","bundle":{"reason":"protocol_error","details":' +
                                         json.dumps({"msg": str(e)}) + "}}")
                    continue
                try:
                    new_op, out_bundle = await self.handler(op, bundle)
                    envelope = self.sink.guard_outbound({"op": new_op, "bundle": out_bundle})
                    await conn.send_text(pack(envelope["op"], envelope["bundle"]).decode("utf-8"))
                except Reject as r:
                    await conn.send_text(pack("control/error", {"reason": r.reason, "details": r.details}).decode("utf-8"))
                except Exception as e:
                    await conn.send_text(pack("control/error", {"reason": "server_exception",
                                                                "details": {"msg": str(e)}}).decode("utf-8"))
        except Exception:
            pass
        finally:
            await conn.close()

    async def start(self):
        self._server = await asyncio.start_server(self._client, self.host, self.port)

    async def run_forever(self):
        if self._server is None:
            await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()