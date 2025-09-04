# imu_repo/runtime/async_sandbox.py
from __future__ import annotations
from typing import Dict, Any, Tuple
import asyncio, time, ssl

from runtime.metrics import metrics, atimer

class SandboxError(RuntimeError): pass
class PolicyError(SandboxError): pass
class ThrottleExceeded(SandboxError): pass

def _now() -> float: return time.monotonic()

class TokenBucket:
    def __init__(self, rate_per_sec: float, capacity: float | None=None):
        self.rate = float(rate_per_sec)
        self.capacity = float(capacity if capacity is not None else rate_per_sec)
        self.tokens = self.capacity
        self.ts = _now()
        self._lock = asyncio.Lock()
    async def take(self, n: float=1.0) -> None:
        async with self._lock:
            now = _now()
            # מילוי מחדש
            self.tokens = min(self.capacity, self.tokens + (now - self.ts)*self.rate)
            self.ts = now
            if self.tokens >= n:
                self.tokens -= n
                return
            raise ThrottleExceeded("rate_limit")

class SandboxRuntime:
    """
    סנדבוקס אסינכרוני עם:
      - sleep_ms עם מקסימום
      - HTTP GET (לא מוצפן) ע"י asyncio.open_connection (מותאם ל-localhost/127.0.0.1)
      - טוקן-באקט לבקרת TPS
      - allowlist למארחים
    """
    def __init__(self, *, allow_hosts=None, http_tps: float=5.0, max_sleep_ms:int=2000):
        self.allow_hosts = set(allow_hosts or ["127.0.0.1","localhost"])
        self.http_bucket = TokenBucket(http_tps, http_tps)
        self.max_sleep_ms = int(max_sleep_ms)

    async def sleep_ms(self, ms: int) -> None:
        ms = int(ms)
        if ms < 0: ms = 0
        if ms > self.max_sleep_ms:
            raise PolicyError(f"sleep_ms_exceeds_policy:{ms}>{self.max_sleep_ms}")
        async with atimer("sandbox.sleep_ms"):
            await asyncio.sleep(ms/1000.0)

    def _check_host(self, host: str) -> None:
        if host not in self.allow_hosts:
            raise PolicyError(f"host_not_allowed:{host}")

    async def http_get(self, host: str, port: int, path: str="/", *, timeout_s: float=3.0) -> Tuple[int, Dict[str,str], bytes]:
        """
        HTTP/1.1 GET פשוט (ללא TLS) — מיועד ל-local servers בטסטים.
        אוכף allowlist ו-TPS.
        """
        self._check_host(host)
        await self.http_bucket.take(1.0)
        req = f"GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\nUser-Agent: imu-sbx\r\nAccept: */*\r\n\r\n".encode("utf-8")
        async with atimer("sandbox.http_get"):
            reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout_s)
            try:
                writer.write(req)
                await writer.drain()
                data = await asyncio.wait_for(reader.read(-1), timeout=timeout_s)
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass
        # פיענוח כותרות
        header, _, body = data.partition(b"\r\n\r\n")
        status = 0
        headers: Dict[str,str] = {}
        try:
            lines = header.split(b"\r\n")
            if lines:
                parts = lines[0].split()
                if len(parts)>=2 and parts[1].isdigit():
                    status = int(parts[1])
            for ln in lines[1:]:
                if b":" in ln:
                    k,v = ln.split(b":",1)
                    headers[k.decode("latin1").strip().lower()] = v.decode("latin1").strip()
        except Exception:
            # לא מפיל — זה סנדבוקס; מחזיר raw
            pass
        metrics.inc("sandbox.http_get.count", 1)
        return status, headers, body