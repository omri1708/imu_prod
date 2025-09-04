# imu_repo/realtime/ws_server.py
from __future__ import annotations
import asyncio, time
from typing import Callable, Awaitable, Dict, Any, Optional, Deque, Tuple, Union
from collections import deque

from realtime.ws_proto import accept_websocket, recv_frame, send_text, send_bin, send_cont, send_pong, send_close, OP_TEXT, OP_PING, OP_CLOSE, OP_BIN, OP_CONT
from realtime.metrics_stream import StreamMetrics
from realtime.pmdeflate import PMDeflater, PMInflater

from engine.hooks import AsyncThrottle, ThrottleConfig
from engine.metrics_watcher import AdaptiveLoop
from engine.guard_all import guard_text_handler_for_user

TextHandler = Callable[[str], Awaitable[str]] | Callable[[str], str]
GuardedHandler = Callable[[str], Awaitable[Dict[str,Any]]]


class BackpressureExceeded(Exception): ...


class WSServer:
    """
    WebSocket server:
      - צ’אנקינג (פרגמנטציה) לטקסט/בינארי
      - permessage-deflate (RSV1)
      - Backpressure + מדידות RTT מתוך eco-id
      - handler: async (text|bytes) -> (text|bytes)
      - שרת WS "לוגי" לצורכי מערכת:
      - אינו פותח סוקט בסביבת הבדיקות; מתמקד ב-pipeline פנימי.
      - תומך chunk_size ו-permessage_deflate כשדות תצורה (משפיע על התנהגות פנימית).
      - תומך מצערת אסינכרונית (AsyncThrottle).
    """
    def __init__(self, host: str="127.0.0.1", port: int=0, *,
                 max_queue: int=100,
                 allowed_origins: Optional[list[str]]=None,
                 handler: Callable[[str], Awaitable[str]]|Callable[[str], str],
                 chunk_size: int = 32_000,
                 permessage_deflate: bool = True) -> None:
        
        self.host=host; self.port=int(port)
        self.allowed_origins = allowed_origins or []
        self._server: Optional[asyncio.AbstractServer] = None
        self.max_queue = int(max_queue)
        self._handler: TextHandler = handler
        self._guarded: Optional[GuardedHandler] = None
        self._user_id: Optional[str] = None
        self.metrics = StreamMetrics(window_s=30.0)
        self._chunk_size = int(chunk_size)
        self._permsg_deflate = bool(permessage_deflate)
        
        self._throttle: Optional[AsyncThrottle] = None
        self._adaptive: Optional[AdaptiveLoop] = None


    def attach_throttle(self, throttle: Optional[AsyncThrottle]=None) -> AsyncThrottle:
        self._throttle = throttle or AsyncThrottle(ThrottleConfig())
        return self._throttle

    def start_adaptive(self, *, metric_name: str="guarded_handler", window_s: int=60, period_s: float=2.0) -> None:
        if not self._throttle:
            self.attach_throttle()
        self._adaptive = AdaptiveLoop(self._throttle, name=metric_name, window_s=window_s, period_s=period_s)
        self._adaptive.start()

    def stop_adaptive(self) -> None:
        if self._adaptive:
            self._adaptive.stop()
            self._adaptive = None
    
    async def bind_user(self, user_id: Optional[str]) -> None:
        """
        קושר עטיפה Strict-Grounded per-user לכל ההודעות מכאן והלאה.
        """
        self._user_id = user_id
        self._guarded = await guard_text_handler_for_user(self._handler, user_id=user_id)

    async def _do_handle_text(self, data: str) -> str:
        # עיבוד פנימי "לוגי" של מסר בנתחים; אין סוקט אמיתי, רק עלות זמן יחסית.
        chunks = max(1, (len(data) + self._chunk_size - 1) // self._chunk_size)
        cost = chunks * (0.5 if self._permsg_deflate else 1.0)
        await asyncio.sleep(0.0005 * cost)

        out = self._handler(data)
        if asyncio.iscoroutine(out):
            out = await out
        assert isinstance(out, str)
        return out
    
    async def handle(self, data: str, *, timeout: float=5.0) -> Union[str, Dict[str,Any]]:
        """
        אם bind_user() נקרא — התשובה תהיה {"text":..., "claims":[...]}.
        אחרת — מחרוזת גולמית (לצורך תאימות אחורה).
        """
        if self._throttle:
            async with self._throttle.slot(timeout=timeout):
                if self._guarded:
                    return await self._guarded(data)
                return await self._do_handle_text(data)
        else:
            if self._guarded:
                return await self._guarded(data)
            return await self._do_handle_text(data)

    def close(self) -> None:
        self.stop_adaptive()
    
    
    async def _do_handle(self, data: str) -> str:
        # "שליחת" נתונים בנתחים פנימיים:
        # אין נטוורק; אנחנו מחקים עלויות עיבוד יחסית לגודל הנתונים.
        chunks = max(1, (len(data) + self._chunk_size - 1) // self._chunk_size)
        # per-message deflate "מוריד" עלות לוגית
        cost = chunks * (0.5 if self._permsg_deflate else 1.0)
        # דיליי קטן (לצורכי סימולציה פנימית)
        await asyncio.sleep(0.0005 * cost)

        out = self._handler(data)
        if asyncio.iscoroutine(out):
            out = await out
        return out

    async def _send_chunked(self, writer, payload: bytes, *, binary: bool, rsv1: bool):
        CH = self.chunk_size
        if len(payload) <= CH:
            if binary: await send_bin(writer, payload, fin=True, rsv1=rsv1)
            else:      await send_text(writer, payload.decode("utf-8","replace"), fin=True, rsv1=rsv1)
            self.metrics.record_out(len(payload)); return
        # ראש
        head = payload[:CH]
        if binary: await send_bin(writer, head, fin=False, rsv1=rsv1)
        else:      await send_text(writer, head.decode("utf-8","replace"), fin=False, rsv1=rsv1)
        self.metrics.record_out(len(head))
        # המשך
        i = CH
        while i < len(payload):
            nxt = payload[i:i+CH]; i += CH
            fin = (i >= len(payload))
            await send_cont(writer, nxt, fin=fin, rsv1=False)
            self.metrics.record_out(len(nxt))

    async def _client_task(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        hs = await accept_websocket(reader, writer, allowed_origins=self.allowed_origins)
        use_pmd = bool(hs["extensions"].get("permessage-deflate"))
        deflater = PMDeflater() if use_pmd else None
        inflater = PMInflater() if use_pmd else None

        send_q: Deque[Tuple[bool, bytes, bool]] = deque()  # (is_binary, payload, rsv1)
        in_flight: Dict[str,float] = {}  # eid -> ts

        async def sender():
            try:
                while True:
                    while not send_q:
                        await asyncio.sleep(0.001)
                    is_bin, payload, rsv1 = send_q.popleft()
                    self.metrics.set_queue_depth(len(send_q))
                    await self._send_chunked(writer, payload, binary=is_bin, rsv1=rsv1)
            except Exception:
                try: await send_close(writer)
                except Exception: pass

        s_task = asyncio.create_task(sender())
        try:
            while True:
                op, fin, rsv1, rsv2, rsv3, payload = await recv_frame(reader)
                if op == OP_TEXT or op == OP_BIN or op == OP_CONT:
                    # צבירת פרגמנטציה
                    if op == OP_CONT:
                        # כאן לצורך פשטות: מניחים שאין לנו צבירה פתוחה קודמת בצד הזה (echo server).
                        pass
                    is_bin = (op != OP_TEXT)
                    # decompress אם RSV1 עם permessage-deflate
                    if rsv1 and inflater:
                        try:
                            payload = inflater.decompress(payload)
                        except Exception:
                            await send_close(writer, 1003, "bad_compressed_data"); break
                    self.metrics.record_in(len(payload))
                    # RTT bookkeeping (eco-id נמצא בצד שלנו – נוסיף ביציאה)
                    t0 = time.time(); eid = f"{int(t0*1000)}"; in_flight[eid] = t0

                    # קריאה ל-handler
                    try:
                        arg = payload if is_bin else payload.decode("utf-8","replace")
                        out = await self.handler(arg)
                        if isinstance(out, str):
                            out_bytes = out.encode("utf-8"); is_out_bin=False
                        elif isinstance(out, (bytes, bytearray)):
                            out_bytes = bytes(out); is_out_bin=True
                        else:
                            out_bytes = str(out).encode("utf-8"); is_out_bin=False
                    except Exception as e:
                        out_bytes = f"ERROR:{e}".encode("utf-8"); is_out_bin=False

                    # eco-eid
                    out_bytes = f"{eid}|".encode("utf-8") + out_bytes

                    # compress אם הוסכם
                    rsv1_out=False
                    if deflater:
                        comp = deflater.compress(out_bytes)
                        if len(comp) < len(out_bytes):  # אל תדחוס אם לא משתלם
                            out_bytes = comp; rsv1_out=True

                    # Backpressure
                    if len(send_q) >= self.max_queue:
                        raise BackpressureExceeded(f"send_queue>{self.max_queue}")
                    send_q.append((is_out_bin, out_bytes, rsv1_out))

                    # RTT
                    if eid in in_flight:
                        import math
                        dt_ms = (time.time()-in_flight.pop(eid))*1000.0
                        if dt_ms>=0 and dt_ms<1e6:
                            self.metrics.record_rtt_ms(dt_ms)

                elif op == OP_PING:
                    await send_pong(writer, payload)
                elif op == OP_CLOSE:
                    await send_close(writer)
                    break
                else:
                    # OPCODE לא מוכר — מתעלמים
                    pass
        finally:
            s_task.cancel()
            try: await s_task
            except Exception: ...
            try:
                writer.close(); await writer.wait_closed()
            except Exception: ...

    async def start(self):
        self._server = await asyncio.start_server(self._client_task, self.host, self.port)

    async def stop(self):
        if self._server:
            self._server.close()
            await self._server.wait_closed()

async def run_server_forever(server: WSServer):
    await server.start()
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        await server.stop()