# imu_repo/realtime/tcp_framed.py
from __future__ import annotations
import asyncio, struct
from typing import Callable, Awaitable, Dict, Any, Optional, Tuple
from realtime.protocol import pack, unpack, ProtocolError
from realtime.strict_sink import StrictSink, Reject

Handler = Callable[[str, Dict[str, Any]], Awaitable[Tuple[str, Dict[str, Any]]]]

class TCPFramedServer:
    """
    פרוטוקול: [uint32 BE length][utf-8 json]
    ה-json הוא {"op": "...", "bundle": {...}} לפי realtime.protocol
    """
    def __init__(self, host: str, port: int, handler: Handler, sink: StrictSink):
        self.host = host
        self.port = port
        self.handler = handler
        self.sink = sink
        self._server: Optional[asyncio.base_events.Server] = None

    async def _send(self, writer: asyncio.StreamWriter, payload: bytes):
        writer.write(struct.pack(">I", len(payload)))
        writer.write(payload)
        await writer.drain()

    async def _recv(self, reader: asyncio.StreamReader) -> bytes:
        hdr = await reader.readexactly(4)
        (n,) = struct.unpack(">I", hdr)
        if n > 16 * 1024 * 1024:
            raise ProtocolError("message too large")
        return await reader.readexactly(n)

    async def handle_conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        peer = writer.get_extra_info("peername")
        try:
            # ברכת פתיחה בקרה (ללא claims)
            await self._send(writer, pack("control/hello", {"server": "tcp_framed", "ok": True}))

            while True:
                data = await self._recv(reader)
                try:
                    op, bundle = unpack(data)
                except ProtocolError as e:
                    await self._send(writer, pack("control/error", {"reason": "protocol_error", "detail": str(e)}))
                    continue

                try:
                    # מעבדים בלוגיקה העסקית (יכול להיות long-running/async)
                    new_op, out_bundle = await self.handler(op, bundle)
                    # *לפני* שליחה: StrictSink מוודא שאין זליגה של תוכן בלי claims+evidence
                    envelope = self.sink.guard_outbound({"op": new_op, "bundle": out_bundle})
                    await self._send(writer, pack(envelope["op"], envelope["bundle"]))
                except Reject as r:
                    await self._send(writer, pack("control/error", {"reason": r.reason, "details": r.details}))
                except Exception as e:
                    await self._send(writer, pack("control/error", {"reason": "server_exception", "details": {"msg": str(e)}}))
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def start(self):
        self._server = await asyncio.start_server(self.handle_conn, self.host, self.port)

    async def run_forever(self):
        if self._server is None:
            await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()