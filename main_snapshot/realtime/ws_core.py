# imu_repo/realtime/ws_core.py
from __future__ import annotations
import asyncio, base64, hashlib, os, struct, zlib
from typing import Tuple, Optional

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

# -----------------------------
# Handshake
# -----------------------------
async def handshake_server(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> dict:
    # קריאת בקשת HTTP
    data = await reader.readuntil(b"\r\n\r\n")
    req = data.decode("latin1", "ignore").split("\r\n")
    headers = {}
    for line in req[1:]:
        if not line: continue
        if ":" in line:
            k,v = line.split(":",1)
            headers[k.strip().lower()] = v.strip()
    key = headers.get("sec-websocket-key")
    if not key:
        writer.close()
        await writer.wait_closed()
        raise ValueError("no_sec_websocket_key")

    accept = base64.b64encode(hashlib.sha1((key + GUID).encode("ascii")).digest()).decode("ascii")
    # ניהול הרחבה permessage-deflate: אם הלקוח ביקש — נאשר (ללא פרמטרים מתקדמים)
    extensions = headers.get("sec-websocket-extensions","")
    enable_pmd = "permessage-deflate" in extensions.lower()

    resp = [
        "HTTP/1.1 101 Switching Protocols",
        "Upgrade: websocket",
        "Connection: Upgrade",
        f"Sec-WebSocket-Accept: {accept}",
    ]
    if enable_pmd:
        resp.append("Sec-WebSocket-Extensions: permessage-deflate")
    resp.append("\r\n")
    writer.write(("\r\n".join(resp)).encode("latin1"))
    await writer.drain()
    return {"permessage_deflate": enable_pmd}

async def handshake_client(host: str, port: int, path: str = "/", enable_pmd: bool = True) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter, dict]:
    reader, writer = await asyncio.open_connection(host, port)
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    hdrs = [
        f"GET {path} HTTP/1.1",
        f"Host: {host}:{port}",
        "Upgrade: websocket",
        "Connection: Upgrade",
        f"Sec-WebSocket-Key: {key}",
        "Sec-WebSocket-Version: 13",
    ]
    if enable_pmd:
        hdrs.append("Sec-WebSocket-Extensions: permessage-deflate")
    hdrs.append("\r\n")
    writer.write(("\r\n".join(hdrs)).encode("latin1"))
    await writer.drain()

    data = await reader.readuntil(b"\r\n\r\n")
    res = data.decode("latin1", "ignore").split("\r\n")
    status = res[0]
    if "101" not in status:
        raise ValueError(f"handshake_failed: {status}")
    headers = {}
    for line in res[1:]:
        if not line: continue
        if ":" in line:
            k,v = line.split(":",1)
            headers[k.strip().lower()] = v.strip()
    enable_pmd_ok = "permessage-deflate" in headers.get("sec-websocket-extensions","").lower()
    return reader, writer, {"permessage_deflate": enable_pmd_ok}

# -----------------------------
# Frames: encode/decode
# -----------------------------
OP_CONT = 0x0
OP_TEXT = 0x1
OP_BIN  = 0x2
OP_CLOSE= 0x8
OP_PING = 0x9
OP_PONG = 0xA

PMD_TAIL = b"\x00\x00\xff\xff"  # RFC7692 tail for raw DEFLATE stream end

def _mask_bytes(data: bytes, mask_key: bytes) -> bytes:
    return bytes(b ^ mask_key[i % 4] for i,b in enumerate(data))

def _pack_frame(opcode: int, payload: bytes, *, mask: bool, compressed: bool) -> bytes:
    fin = 0x80
    rsv1 = 0x40 if compressed else 0x00
    b0 = fin | rsv1 | (opcode & 0x0F)
    # length encoding
    n = len(payload)
    if n < 126:
        b1 = (0x80 if mask else 0x00) | n
        header = bytes([b0, b1])
        ext = b""
    elif n <= 0xFFFF:
        b1 = (0x80 if mask else 0x00) | 126
        header = bytes([b0, b1])
        ext = struct.pack("!H", n)
    else:
        b1 = (0x80 if mask else 0x00) | 127
        header = bytes([b0, b1])
        ext = struct.pack("!Q", n)
    if mask:
        mkey = os.urandom(4)
        payload = _mask_bytes(payload, mkey)
        return header + ext + mkey + payload
    else:
        return header + ext + payload

def encode_text(s: str, *, client: bool, permessage_deflate: bool) -> bytes:
    data = s.encode("utf-8")
    compressed = False
    if permessage_deflate and len(data) > 0:
        comp = zlib.compressobj(wbits=-zlib.MAX_WBITS)
        data = comp.compress(data) + comp.flush(zlib.Z_SYNC_FLUSH)
        if data.endswith(PMD_TAIL):
            data = data[:-4]
        compressed = True
    return _pack_frame(OP_TEXT, data, mask=client, compressed=compressed)

def encode_bin(b: bytes, *, client: bool, permessage_deflate: bool) -> bytes:
    data = b
    compressed = False
    if permessage_deflate and len(data) > 0:
        comp = zlib.compressobj(wbits=-zlib.MAX_WBITS)
        data = comp.compress(data) + comp.flush(zlib.Z_SYNC_FLUSH)
        if data.endswith(PMD_TAIL):
            data = data[:-4]
        compressed = True
    return _pack_frame(OP_BIN, data, mask=client, compressed=compressed)

async def read_frame(reader: asyncio.StreamReader, *, server_side: bool, permessage_deflate: bool) -> Tuple[int, bytes, bool]:
    # returns (opcode, payload_bytes, compressed_flag)
    b0 = await reader.readexactly(1)
    b1 = await reader.readexactly(1)
    fin = (b0[0] & 0x80) != 0
    rsv1 = (b0[0] & 0x40) != 0
    opcode = b0[0] & 0x0F
    masked = (b1[0] & 0x80) != 0
    ln = (b1[0] & 0x7F)
    if ln == 126:
        ext = await reader.readexactly(2)
        ln = struct.unpack("!H", ext)[0]
    elif ln == 127:
        ext = await reader.readexactly(8)
        ln = struct.unpack("!Q", ext)[0]
    mkey = b""
    if masked:
        mkey = await reader.readexactly(4)
    data = await reader.readexactly(ln)
    if masked:
        data = _mask_bytes(data, mkey)
    compressed = bool(rsv1 and permessage_deflate and opcode in (OP_TEXT, OP_BIN))
    if compressed and len(data) > 0:
        # RFC7692: add tail
        data = data + PMD_TAIL
        decomp = zlib.decompressobj(wbits=-zlib.MAX_WBITS)
        data = decomp.decompress(data) + decomp.flush()
    return opcode, data, compressed