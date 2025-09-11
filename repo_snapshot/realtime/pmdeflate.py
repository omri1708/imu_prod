# imu_repo/realtime/pmdeflate.py
from __future__ import annotations
import zlib

# raw DEFLATE (wbits=-15) + סיום מסר (RFC 7692: הוספת 0x00 0x00 0xff 0xff)
_TRAILER = b"\x00\x00\xff\xff"

class PMDeflater:
    def __init__(self):
        self.c = zlib.compressobj(wbits=-15)

    def compress(self, data: bytes) -> bytes:
        out = self.c.compress(data) + self.c.flush(zlib.Z_SYNC_FLUSH)
        # הסר 0x00 0x00 ff ff בסוף (כמתחייב מהרחבה)
        if out.endswith(_TRAILER):
            out = out[:-4]
        return out

class PMInflater:
    def __init__(self):
        self.d = zlib.decompressobj(wbits=-15)

    def decompress(self, data: bytes) -> bytes:
        # הוסף טריילר בסוף כדי לסמן סוף־מסר
        return self.d.decompress(data + _TRAILER)