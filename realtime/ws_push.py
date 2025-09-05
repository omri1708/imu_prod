# imu_repo/realtime/ws_push.py
from __future__ import annotations
import asyncio, base64, hashlib, json, struct
from typing import Dict, Any, Tuple, Set

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

class WSError(Exception): pass

class WSConnection:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.r = reader
        self.w = writer
        self.alive = True

    async def handshake(self):
        # קריאת בקשת HTTP
        req = await self.r.readuntil(b"\r\n\r\n")
        try:
            header = req.decode("utf-8", "ignore")
            lines = header.split("\r\n")
            first = lines[0]
            if "Upgrade: websocket" not in header and "upgrade: websocket" not in header:
                raise WSError("not a WS upgrade")
            key = None
            for ln in lines[1:]:
                if ln.lower().startswith("sec-websocket-key:"):
                    key = ln.split(":", 1)[1].strip()
                    break
            if not key:
                raise WSError("no Sec-WebSocket-Key")
            acc = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
            resp = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {acc}\r\n"
                "\r\n"
            ).encode("utf-8")
            self.w.write(resp)
            await self.w.drain()
        except Exception as e:
            raise WSError(f"handshake failed: {e}")

    async def recv_text(self) -> Tuple[str, Dict[str, Any]]:
        # תמיכה בסיסית ב־Opcode=1 (טקסט), מסגרות בודדות
        hdr = await self.r.readexactly(2)
        b1, b2 = hdr[0], hdr[1]
        fin = (b1 & 0x80) != 0
        opcode = b1 & 0x0F
        masked = (b2 & 0x80) != 0
        ln = (b2 & 0x7F)
        if ln == 126:
            ext = await self.r.readexactly(2)
            ln = struct.unpack(">H", ext)[0]
        elif ln == 127:
            ext = await self.r.readexactly(8)
            ln = struct.unpack(">Q", ext)[0]
        mask = b""
        if masked:
            mask = await self.r.readexactly(4)
        payload = await self.r.readexactly(ln)
        if masked:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        if opcode == 0x8:  # close
            self.alive = False
            return "control/close", {}
        if opcode != 0x1:  # text only here
            return "control/ignore", {}
        try:
            doc = json.loads(payload.decode("utf-8"))
            return doc.get("op", "msg"), doc.get("bundle", {})
        except Exception:
            return "control/error", {"reason": "bad_json"}

    async def send_text(self, text: str):
        data = text.encode("utf-8")
        b1 = 0x80 | 0x1
        ln = len(data)
        if ln < 126:
            header = struct.pack("!BB", b1, ln)
        elif ln < (1 << 16):
            header = struct.pack("!BBH", b1, 126, ln)
        else:
            header = struct.pack("!BBQ", b1, 127, ln)
        self.w.write(header + data)
        await self.w.drain()

    async def send_json(self, op: str, bundle: Dict[str, Any]):
        await self.send_text(json.dumps({"op": op, "bundle": bundle}))

    def close_now(self):
        if not self.w.is_closing():
            try:
                self.w.close()
            except Exception:
                pass
        self.alive = False


class WSPushServer:
    """
    שרת WS עם רישום כל החיבורים ו-Broadcast לכולם.
    משלב Grounded-Strict בצד השרת באמצעות sink חיצוני (StrictSink).
    """
    def __init__(self, host: str, port: int, handler, sink):
        self.host = host
        self.port = port
        self._handler = handler  # async (op,bundle) -> (op_out, bundle_out) or raises
        self._sink = sink        # must expose: policy + validate_and_wrap(op,bundle)->(op,bundle) or error
        self._srv: asyncio.AbstractServer | None = None
        self._conns: Set[WSConnection] = set()
        self._lock = asyncio.Lock()

    async def _client_task(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        conn = WSConnection(reader, writer)
        try:
            await conn.handshake()
            async with self._lock:
                self._conns.add(conn)
            # hello
            await conn.send_json("control/hello", {"policy": getattr(self._sink, "policy", {})})
            while conn.alive:
                try:
                    op, bundle = await conn.recv_text()
                except asyncio.IncompleteReadError:
                    break
                if not conn.alive: break
                if op in ("control/close", "control/ignore"): 
                    continue
                # מיישמים רגולציה/אימות בסינק
                ok, op2, bundle2 = await self._sink.try_accept(op, bundle)
                if not ok:
                    await conn.send_json("control/error", bundle2)  # bundle2 = reason
                    continue
                # מעבירים ל־handler האפליקטיבי
                op_out, bundle_out = await self._handler(op2, bundle2)
                # גם יציאה נעטפת במדיניות Grounded
                ok2, op3, bundle3 = await self._sink.try_accept(op_out, bundle_out)
                if not ok2:
                    await conn.send_json("control/error", bundle3)
                    continue
                await conn.send_json(op3, bundle3)
        except Exception as e:
            try:
                await conn.send_json("control/error", {"reason": f"{e}"})
            except Exception:
                pass
        finally:
            conn.close_now()
            async with self._lock:
                self._conns.discard(conn)

    async def start(self):
        self._srv = await asyncio.start_server(self._client_task, self.host, self.port)

    async def run_forever(self):
        assert self._srv is not None
        async with self._srv:
            print(f"WS Push server at {self.host}:{self.port}")
            await self._srv.serve_forever()

    async def broadcast(self, op: str, bundle: Dict[str, Any]):
        # Grounded-Strict גם על שידור יזום
        ok, op2, bundle2 = await self._sink.try_accept(op, bundle)
        if not ok:
            return
        # שליחה לכל החיבורים הקיימים
        async with self._lock:
            conns = list(self._conns)
        for c in conns:
            if not c.alive: 
                continue
            try:
                await c.send_json(op2, bundle2)
            except Exception:
                c.close_now()