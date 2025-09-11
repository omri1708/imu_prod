# imu_repo/realtime/ws_loopback.py
from __future__ import annotations
import asyncio, time
from typing import Optional, AsyncIterator
from grounded.claims import current
from engine.policy_ctx import get_user
from engine.config import load_config
from realtime.ws_core import handshake_server, handshake_client, encode_bin, read_frame, OP_BIN, OP_TEXT, OP_CLOSE

class Credit:
    def __init__(self, initial: int):
        self._avail = initial
        self._cond = asyncio.Condition()

    async def acquire(self) -> None:
        async with self._cond:
            while self._avail <= 0:
                await self._cond.wait()
            self._avail -= 1

    async def grant(self, n: int = 1) -> None:
        async with self._cond:
            self._avail += n
            self._cond.notify_all()

def _rt_cfg():
    cfg = load_config()
    r = dict(cfg.get("realtime", {}))
    r.setdefault("chunk_bytes", 32 * 1024)
    r.setdefault("initial_credits", 4)
    r.setdefault("permessage_deflate", True)
    cfg["realtime"] = r
    return r

async def start_loopback_server(host: str = "127.0.0.1", port: int = 0):
    """
    שרת WS לוקאלי: פרוטוקול echo עם back-pressure.
    כל הודעת נתונים שמתקבלת -> נרשמת Evidence, ואז נשלחת חזרה (echo),
    ורק לאחר העיבוד — מעניקים CREDIT להודעה הבאה (שומר על back-pressure).
    """
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        info = await handshake_server(reader, writer)
        pmd = bool(info.get("permessage_deflate", False))
        uid = get_user() or "anon"
        current().add_evidence("ws_open", {"source_url": f"ws://{host}:{port}", "trust": 0.95, "ttl_s": 600, "payload": {"user": uid, "pmd": pmd}})
        try:
            while True:
                op, payload, compressed = await read_frame(reader, server_side=True, permessage_deflate=pmd)
                if op == OP_CLOSE:
                    break
                if op in (OP_TEXT, OP_BIN):
                    # Evidence קבלה
                    current().add_evidence("ws_recv", {
                        "source_url": f"ws://{host}:{port}", "trust": 0.95, "ttl_s": 600,
                        "payload": {"bytes": len(payload), "compressed": compressed}
                    })
                    # Echo חזרה
                    frame = encode_bin(payload, client=False, permessage_deflate=pmd)
                    writer.write(frame)
                    await writer.drain()
                    # מעניקים קרדיט (מסמנים לקליינט שהוא יכול לשלוח עוד צ'אנק)
                    credit_msg = b"CREDIT:1"
                    writer.write(encode_bin(credit_msg, client=False, permessage_deflate=False))
                    await writer.drain()
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            current().add_evidence("ws_close", {"source_url": f"ws://{host}:{port}", "trust": 0.95, "ttl_s": 600, "payload": {"user": uid}})
    srv = await asyncio.start_server(handle, host, port)
    sock = next(iter(srv.sockets))
    h, p = sock.getsockname()
    return srv, h, p

class WSClient:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, *, pmd: bool, chunk: int, credits: int):
        self.reader = reader
        self.writer = writer
        self.pmd = pmd
        self.chunk = int(chunk)
        self.credit = Credit(credits)

    async def _read_loop_credit(self):
        # מאזין למסרים שמחזירים CREDIT או ECHO, ומזרים Evidences
        while True:
            try:
                op, payload, compressed = await read_frame(self.reader, server_side=False, permessage_deflate=self.pmd)
            except asyncio.IncompleteReadError:
                break
            if op == OP_CLOSE:
                break
            if payload.startswith(b"CREDIT:"):
                # מעניק אשראי לשליחה
                try:
                    n = int(payload.split(b":",1)[1])
                except Exception:
                    n = 1
                await self.credit.grant(n)
            else:
                # Evidence קבלה (echo)
                current().add_evidence("ws_echo", {
                    "source_url": "local://ws_loopback", "trust": 0.96, "ttl_s": 600,
                    "payload": {"bytes": len(payload), "compressed": compressed}
                })

    async def send_bytes(self, b: bytes) -> None:
        # צ'אנקינג + back-pressure: כל צ'אנק דורש קרדיט
        for i in range(0, len(b), self.chunk):
            await self.credit.acquire()
            part = b[i:i+self.chunk]
            frame = encode_bin(part, client=True, permessage_deflate=self.pmd)
            self.writer.write(frame)
            await self.writer.drain()
            current().add_evidence("ws_send", {"source_url":"local://ws_loopback","trust":0.96,"ttl_s":600,"payload":{"bytes": len(part), "compressed": self.pmd}})

    async def close(self):
        # שליחת CLOSE פשוטה (ללא קוד סיבה)
        self.writer.write(b"\x88\x00")
        await self.writer.drain()
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except Exception:
            pass

async def open_loopback_client(host: str, port: int) -> WSClient:
    cfg = _rt_cfg()
    reader, writer, info = await handshake_client(host, port, enable_pmd=bool(cfg["permessage_deflate"]))
    cli = WSClient(reader, writer, pmd=bool(info.get("permessage_deflate", False)),
                   chunk=int(cfg["chunk_bytes"]), credits=int(cfg["initial_credits"]))
    # מריצים קורא אשראי ברקע
    asyncio.create_task(cli._read_loop_credit())
    return cli