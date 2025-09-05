import asyncio, base64, hashlib, struct, zlib
from typing import Dict, Any
from .priority_bus import AsyncPriorityTopicBus
from .backpressure import GlobalTokenBucket

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

class WSProtocolError(Exception): pass

class WebSocketConnection:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, bus: AsyncPriorityTopicBus):
        self.r = reader
        self.w = writer
        self.bus = bus
        self.alive = True
        self.topics = set()
        self.compressor = None  # per-message deflate (optional)
        self.decompressor = None

    async def handshake(self):
        data = await self.r.readuntil(b"\r\n\r\n")
        headers = data.decode("utf-8", "ignore").split("\r\n")
        req = headers[0]
        hdrs = {}
        for h in headers[1:]:
            if ":" in h:
                k, v = h.split(":", 1)
                hdrs[k.strip().lower()] = v.strip()
        if "upgrade" not in hdrs.get("connection", "").lower():
            raise WSProtocolError("no upgrade")
        key = hdrs.get("sec-websocket-key")
        if not key:
            raise WSProtocolError("no key")
        accept = base64.b64encode(hashlib.sha1((key+GUID).encode()).digest()).decode()
        # permessage-deflate (אופציונלי; נתמוך בדחיסה יוצאת)
        ext = hdrs.get("sec-websocket-extensions", "")
        use_deflate = "permessage-deflate" in ext
        resp = [
            "HTTP/1.1 101 Switching Protocols",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Accept: {accept}",
        ]
        if use_deflate:
            resp.append("Sec-WebSocket-Extensions: permessage-deflate")
            self.compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
            self.decompressor = zlib.decompressobj(wbits=-zlib.MAX_WBITS)
        resp_bytes = ("\r\n".join(resp) + "\r\n\r\n").encode()
        self.w.write(resp_bytes)
        await self.w.drain()

    async def recv_frame(self) -> bytes:
        hdr = await self.r.readexactly(2)
        b1, b2 = hdr[0], hdr[1]
        fin = (b1 >> 7) & 1
        opcode = b1 & 0x0F
        masked = (b2 >> 7) & 1
        length = b2 & 0x7F
        if length == 126:
            length = struct.unpack("!H", await self.r.readexactly(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", await self.r.readexactly(8))[0]
        mask = await self.r.readexactly(4) if masked else b"\x00\x00\x00\x00"
        payload = bytearray(await self.r.readexactly(length))
        if masked:
            for i in range(length):
                payload[i] ^= mask[i % 4]
        if opcode == 0x8:  # close
            self.alive = False
            return b""
        if opcode not in (0x1, 0x2):  # text/binary only כאן
            return b""
        data = bytes(payload)
        # RSV1 -> deflate; בפשטות נתעלם בכניסה (הדגמתית) אם אין דחיסה פעילה
        return data

    async def send_text(self, data: str):
        raw = data.encode()
        await self._send_frame(0x1, raw)

    async def send_binary(self, data: bytes):
        await self._send_frame(0x2, data)

    async def _send_frame(self, opcode: int, payload: bytes):
        # דחיסה יוצאת אם קיימת
        if self.compressor:
            payload = self.compressor.compress(payload) + self.compressor.flush(zlib.Z_SYNC_FLUSH)
            # strip 0x00 0x00 0xff 0xff זנב? (פשטות: משאירים)
        b1 = 0x80 | opcode
        length = len(payload)
        if length < 126:
            hdr = struct.pack("!BB", b1, length)
        elif length < (1 << 16):
            hdr = struct.pack("!BBH", b1, 126, length)
        else:
            hdr = struct.pack("!BBQ", b1, 127, length)
        self.w.write(hdr + payload)
        await self.w.drain()

    async def handle(self):
        await self.handshake()
        # פרוטוקול אפליקטיבי פשוט:
        # SUB topic\n  | PUB topic priority data\n
        # שידורי push: צרכנים על topics שנרשמו.
        async def sender_loop():
            tasks = []
            async def fanout(topic):
                async for payload in self.bus.subscribe(topic):
                    if not self.alive: break
                    # payload יכול להיות str/bytes
                    if isinstance(payload, bytes):
                        await self.send_binary(payload)
                    else:
                        await self.send_text(str(payload))
            for t in list(self.topics):
                tasks.append(asyncio.create_task(fanout(t)))
            if tasks:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        sender_task = asyncio.create_task(sender_loop())
        try:
            while self.alive:
                msg = await self.recv_frame()
                if not msg:
                    break
                try:
                    line = msg.decode().strip()
                except Exception:
                    continue
                if line.startswith("SUB "):
                    topic = line[4:].strip()
                    self.topics.add(topic)
                    # להתעדכן בלולאת השידור
                    sender_task.cancel()
                    sender_task = asyncio.create_task(sender_loop())
                    await self.send_text(f"OK SUB {topic}")
                elif line.startswith("PUB "):
                    try:
                        _, topic, prio_str, rest = line.split(" ", 3)
                        prio = int(prio_str)
                    except ValueError:
                        await self.send_text("ERR bad PUB")
                        continue
                    try:
                        success = await self.bus.publish(topic, rest, priority=prio, cost_tokens=1)
                        await self.send_text("OK PUB" if success else "DROPPED")
                    except Exception as e:
                        await self.send_text(f"ERR {type(e).__name__}: {e}")
                else:
                    await self.send_text("ERR unknown")
        finally:
            self.alive = False
            sender_task.cancel()
            try: await sender_task
            except: pass
            try: self.w.close(); await self.w.wait_closed()
            except: pass

async def run_server(host="0.0.0.0", port=8765):
    global_bucket = GlobalTokenBucket(capacity=5000, rate_tokens_per_sec=1000.0)
    bus = AsyncPriorityTopicBus(global_bucket, per_topic_rates={
        "telemetry": (200, 400.0),
        "logs": (100, 200.0),
        "logic": (500, 800.0),
        "progress:*": (100, 100.0),
    })
    async def _client(reader, writer):
        conn = WebSocketConnection(reader, writer, bus)
        await conn.handle()
    server = await asyncio.start_server(_client, host, port)
    addrs = ", ".join(str(s.getsockname()) for s in server.sockets)
    print(f"WS push server on {addrs}")
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(run_server())