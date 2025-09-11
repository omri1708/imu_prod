# imu_repo/realtime/ws_push.py
from __future__ import annotations
import asyncio, base64, hashlib, json, struct, time
from typing import Dict, Any, Tuple, Set, Optional

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

class WSError(Exception): ...
class RateLimited(Exception): ...

class TokenBucket:
    """Token-bucket פשוט ל- messages/sec ו- bytes/sec."""
    def __init__(self, *, msg_rate: float, byte_rate: float, burst_msgs: int, burst_bytes: int):
        self.msg_rate = msg_rate
        self.byte_rate = byte_rate
        self.burst_msgs = burst_msgs
        self.burst_bytes = burst_bytes
        self._msgs = burst_msgs
        self._bytes = burst_bytes
        self._t = time.monotonic()

    def _replenish(self):
        now = time.monotonic()
        dt = now - self._t
        self._t = now
        self._msgs = min(self.burst_msgs, self._msgs + dt * self.msg_rate)
        self._bytes = min(self.burst_bytes, self._bytes + dt * self.byte_rate)

    def consume(self, n_msgs: int, n_bytes: int) -> bool:
        self._replenish()
        if self._msgs >= n_msgs and self._bytes >= n_bytes:
            self._msgs -= n_msgs
            self._bytes -= n_bytes
            return True
        return False


class WSConnection:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                 *, queue_max: int = 256,
                 bucket: Optional[TokenBucket] = None):
        self.r = reader
        self.w = writer
        self.alive = True
        self.topics: Set[str] = set()
        self._send_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=queue_max)
        # ברירת מחדל: 100 הודעות/שניה, 1MB/s, בורסט 200 הודעות/2MB
        self.bucket = bucket or TokenBucket(msg_rate=100, byte_rate=1_000_000, burst_msgs=200, burst_bytes=2_000_000)
        self._sender_task: Optional[asyncio.Task] = None

    async def start_sender(self):
        async def _pump():
            try:
                while self.alive:
                    frame = await self._send_q.get()
                    if not self.alive: break
                    self.w.write(frame)
                    await self.w.drain()
            except Exception:
                pass
            finally:
                self.close_now()
        self._sender_task = asyncio.create_task(_pump())

    async def enqueue_frame(self, frame: bytes):
        # Back-pressure: אם התור מלא — נמתין עד timeout קצר; אם עדיין מלא, נזרוק הודעה (drop)
        try:
            await asyncio.wait_for(self._send_q.put(frame), timeout=0.250)
        except asyncio.TimeoutError:
            # נזרוק בשקט; האלרם יהיה בטלמטריה של השרת
            pass

    async def handshake(self):
        req = await self.r.readuntil(b"\r\n\r\n")
        header = req.decode("utf-8", "ignore")
        if "upgrade: websocket" not in header.lower():
            raise WSError("not a WS upgrade")
        key = None
        for ln in header.split("\r\n")[1:]:
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
        await self.start_sender()

    async def recv_json(self) -> Tuple[str, Dict[str, Any]]:
        hdr = await self.r.readexactly(2)
        b1, b2 = hdr[0], hdr[1]
        opcode = b1 & 0x0F
        masked = (b2 & 0x80) != 0
        ln = (b2 & 0x7F)
        if ln == 126: ln = struct.unpack(">H", await self.r.readexactly(2))[0]
        elif ln == 127: ln = struct.unpack(">Q", await self.r.readexactly(8))[0]
        mask = b""
        if masked: mask = await self.r.readexactly(4)
        payload = await self.r.readexactly(ln)
        if masked:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        if opcode == 0x8:  # close
            self.alive = False
            return "control/close", {}
        if opcode != 0x1:
            return "control/ignore", {}
        try:
            doc = json.loads(payload.decode("utf-8"))
            return doc.get("op", "msg"), doc.get("bundle", {})
        except Exception:
            return "control/error", {"reason": "bad_json"}

    @staticmethod
    def _frame_text(txt: str) -> bytes:
        data = txt.encode("utf-8")
        b1 = 0x80 | 0x1
        ln = len(data)
        if ln < 126:
            header = struct.pack("!BB", b1, ln)
        elif ln < (1 << 16):
            header = struct.pack("!BBH", b1, 126, ln)
        else:
            header = struct.pack("!BBQ", b1, 127, ln)
        return header + data

    async def send_json(self, op: str, bundle: Dict[str, Any]):
        txt = json.dumps({"op": op, "bundle": bundle})
        frame = self._frame_text(txt)
        # Rate-limit: אם אין אסימון — נזרוק RateLimited והקורא יחליט
        if not self.bucket.consume(1, len(frame)):
            raise RateLimited("ws_send_rate_exceeded")
        await self.enqueue_frame(frame)

    def close_now(self):
        if not self.w.is_closing():
            try:
                self.w.close()
            except Exception:
                pass
        self.alive = False
        if self._sender_task and not self._sender_task.done():
            self._sender_task.cancel()


class WSPushServer:
    """
    שרת WS עם:
      * רישום חיבורים
      * Back-pressure (תור פר-חיבור)
      * Rate-limit (Token-bucket)
      * Subscriptions לנושאים (topics)
      * Grounded-Strict באמצעות sink.try_accept
    """
    def __init__(self, host: str, port: int, handler, sink,
                 *, queue_max=256, msg_rate=100, byte_rate=1_000_000, burst_msgs=200, burst_bytes=2_000_000):
        self.host = host
        self.port = port
        self._handler = handler
        self._sink = sink
        self._srv: asyncio.AbstractServer | None = None
        self._conns: Set[WSConnection] = set()
        self._lock = asyncio.Lock()
        self._queue_max = queue_max
        self._tb_args = dict(msg_rate=msg_rate, byte_rate=byte_rate, burst_msgs=burst_msgs, burst_bytes=burst_bytes)

    async def _client_task(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        conn = WSConnection(reader, writer, queue_max=self._queue_max, bucket=TokenBucket(**self._tb_args))
        try:
            await conn.handshake()
            async with self._lock:
                self._conns.add(conn)
            await conn.send_json("control/hello", {"policy": getattr(self._sink, "policy", {})})
            while conn.alive:
                try:
                    op, bundle = await conn.recv_json()
                except asyncio.IncompleteReadError:
                    break
                if not conn.alive: break
                if op in ("control/close", "control/ignore"):
                    continue

                # ניהול Subscriptions בסיסי
                if op == "ui/subscribe":
                    topics = set(map(str, bundle.get("topics", [])))
                    conn.topics |= topics
                    await conn.send_json("control/ack", {"ok": True, "topics": sorted(conn.topics)})
                    continue
                if op == "ui/unsubscribe":
                    topics = set(map(str, bundle.get("topics", [])))
                    conn.topics -= topics
                    await conn.send_json("control/ack", {"ok": True, "topics": sorted(conn.topics)})
                    continue

                ok, op2, bundle2 = await self._sink.try_accept(op, bundle)
                if not ok:
                    await conn.send_json("control/error", bundle2)
                    continue
                op_out, bundle_out = await self._handler(op2, bundle2)
                ok2, op3, bundle3 = await self._sink.try_accept(op_out, bundle_out)
                if not ok2:
                    await conn.send_json("control/error", bundle3)
                    continue
                try:
                    await conn.send_json(op3, bundle3)
                except RateLimited:
                    # מסמן ללקוח שהורדה נעשתה – לא מפיל חיבור
                    await conn.enqueue_frame(WSConnection._frame_text(json.dumps({
                        "op": "control/warn", "bundle": {"reason": "rate_limited"}
                    })))
        except Exception as e:
            try:
                await conn.enqueue_frame(WSConnection._frame_text(json.dumps({
                    "op": "control/error", "bundle": {"reason": f"{e}"}
                })))
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

    async def broadcast(self, op: str, bundle: Dict[str, Any], *, topic: Optional[str] = None):
        ok, op2, bundle2 = await self._sink.try_accept(op, bundle)
        if not ok:
            return
        async with self._lock:
            conns = list(self._conns)
        for c in conns:
            if not c.alive: 
                continue
            if topic and (topic not in c.topics):
                continue
            try:
                await c.send_json(op2, bundle2)
            except RateLimited:
                # הצב אזהרת rate-limit בלבד
                await c.enqueue_frame(WSConnection._frame_text(json.dumps({
                    "op":"control/warn","bundle":{"reason":"rate_limited"}
                })))
            except Exception:
                c.close_now()