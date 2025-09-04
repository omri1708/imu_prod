# imu_repo/realtime/ws_proto.py
from __future__ import annotations
import asyncio, base64, hashlib, os, struct
from typing import Dict, Any, Tuple, Optional

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

class WSProtocolError(Exception): ...
class WSClosed(Exception): ...

def _b(s: str) -> bytes: return s.encode("utf-8")
def _u(b: bytes) -> str: return b.decode("utf-8", errors="replace")

def _sec_accept(sec_key: str) -> str:
    s = (sec_key + GUID).encode("ascii")
    return base64.b64encode(hashlib.sha1(s).digest()).decode("ascii")

# opcodes
OP_CONT = 0x0
OP_TEXT = 0x1
OP_BIN  = 0x2
OP_CLOSE= 0x8
OP_PING = 0x9
OP_PONG = 0xA

async def accept_websocket(reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                           *, allowed_origins: Optional[list[str]]=None) -> Dict[str,Any]:
    """
    Handshake שרת: מזהה הצעת permessage-deflate ומחזיר flags ב-dict.
    """
    req = await reader.readuntil(b"\r\n\r\n")
    header = _u(req)
    lines = header.split("\r\n")
    if not lines or "GET " not in lines[0]:
        raise WSProtocolError("bad_request_line")
    path = lines[0].split(" ")[1]
    hdrs: Dict[str,str] = {}
    for ln in lines[1:]:
        if ":" in ln:
            k,v = ln.split(":",1); hdrs[k.strip().lower()] = v.strip()
    if hdrs.get("upgrade","").lower()!="websocket" or "upgrade" not in hdrs.get("connection","").lower():
        raise WSProtocolError("no_upgrade")
    if allowed_origins:
        origin = hdrs.get("origin","").lower()
        if origin and origin not in [o.lower() for o in allowed_origins]:
            raise WSProtocolError("origin_blocked")
    key = hdrs.get("sec-websocket-key");    ext_offer = hdrs.get("sec-websocket-extensions","")
    if not key: raise WSProtocolError("no_sec_key")

    # Negotiation: permessage-deflate (no_context_takeover לשני הצדדים)
    use_pmd = False
    if "permessage-deflate" in (ext_offer or ""):
        use_pmd = True

    accept = _sec_accept(key)
    resp = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n"
    )
    if use_pmd:
        resp += "Sec-WebSocket-Extensions: permessage-deflate; server_no_context_takeover; client_no_context_takeover\r\n"
    resp += "\r\n"
    writer.write(_b(resp)); await writer.drain()
    return {"path": path, "headers": hdrs, "extensions": {"permessage-deflate": use_pmd}}

async def _read_exact(reader: asyncio.StreamReader, n: int) -> bytes:
    b = await reader.readexactly(n)
    if not b: raise WSClosed()
    return b

async def recv_frame(reader: asyncio.StreamReader) -> Tuple[int, bool, bool, bool, bool, bytes]:
    """
    מחזיר: (opcode, fin, rsv1, rsv2, rsv3, payload_bytes)
    מסיר מסכה מלקוח.
    """
    b1b2 = await _read_exact(reader, 2)
    b1, b2 = b1b2[0], b1b2[1]
    fin  = (b1 & 0x80) != 0
    rsv1 = (b1 & 0x40) != 0
    rsv2 = (b1 & 0x20) != 0
    rsv3 = (b1 & 0x10) != 0
    opcode = (b1 & 0x0F)
    masked = (b2 & 0x80) != 0
    ln = (b2 & 0x7F)
    if ln==126:
        ln = struct.unpack("!H", await _read_exact(reader, 2))[0]
    elif ln==127:
        ln = struct.unpack("!Q", await _read_exact(reader, 8))[0]
    mask = b""
    if masked:
        mask = await _read_exact(reader, 4)
    payload = await _read_exact(reader, ln) if ln>0 else b""
    if masked and payload:
        payload = bytes(b ^ mask[i%4] for i,b in enumerate(payload))
    return opcode, fin, rsv1, rsv2, rsv3, payload

async def send_frame(writer: asyncio.StreamWriter, opcode: int, payload: bytes=b"", *,
                     fin: bool=True, rsv1: bool=False, rsv2: bool=False, rsv3: bool=False):
    b1 = (0x80 if fin else 0x00) \
        | (0x40 if rsv1 else 0x00) \
        | (0x20 if rsv2 else 0x00) \
        | (0x10 if rsv3 else 0x00) \
        | (opcode & 0x0F)
    ln = len(payload)
    if ln < 126:
        header = struct.pack("!BB", b1, ln)
    elif ln <= 0xFFFF:
        header = struct.pack("!BBH", b1, 126, ln)
    else:
        header = struct.pack("!BBQ", b1, 127, ln)
    writer.write(header + payload)
    await writer.drain()

async def send_text(writer: asyncio.StreamWriter, text: str, *, fin: bool=True, rsv1: bool=False):
    await send_frame(writer, OP_TEXT, _b(text), fin=fin, rsv1=rsv1)

async def send_bin(writer: asyncio.StreamWriter, data: bytes, *, fin: bool=True, rsv1: bool=False):
    await send_frame(writer, OP_BIN, data, fin=fin, rsv1=rsv1)

async def send_cont(writer: asyncio.StreamWriter, data: bytes, *, fin: bool, rsv1: bool=False):
    await send_frame(writer, OP_CONT, data, fin=fin, rsv1=rsv1)

async def send_pong(writer: asyncio.StreamWriter, payload: bytes=b""):
    await send_frame(writer, OP_PONG, payload)

async def send_ping(writer: asyncio.StreamWriter, payload: bytes=b""):
    await send_frame(writer, OP_PING, payload)

async def send_close(writer: asyncio.StreamWriter, code: int=1000, reason: str=""):
    pl = struct.pack("!H", code) + _b(reason)
    await send_frame(writer, OP_CLOSE, pl)
    try: await writer.drain()
    except Exception: pass