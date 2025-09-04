# imu_repo/adapters/net_sandbox.py
from __future__ import annotations
import time, socket, ssl
from urllib.parse import urlparse
from typing import Dict, Any, Optional, Tuple
from grounded.source_policy import policy_singleton as SourcePolicy


class NetError(Exception): ...
class NetDenied(NetError): ...
class NetRateLimit(NetError): ...


class TokenBucket:
    def __init__(self, rps: float, burst: int):
        self.rate = float(rps)
        self.burst = int(burst)
        self.tokens = float(burst)
        self.last = time.time()
    def take(self, cost: float = 1.0) -> bool:
        now = time.time()
        self.tokens = min(self.burst, self.tokens + (now-self.last)*self.rate)
        self.last = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


class NetSandbox:
    """
    Minimal TCP client sandbox:
    - open(host, port) -> conn_id
    - close(conn_id)
    - whitelist of allowed hosts (optional)
    """
    _buckets: Dict[str, TokenBucket] = {}
    def __init__(self, allow_hosts: list[str] | None = None, timeout: float = 5.0):
        self.allow_hosts = set(allow_hosts or [])
        self.timeout = timeout
        self.conns: Dict[str, socket.socket] = {}

    @staticmethod
    def http_get_text(url: str, timeout: float = 2.0) -> str:
        # Allowlist policy
        if not SourcePolicy.domain_allowed(url):
            raise NetDenied(f"domain_not_allowed:{url}")
        u = urlparse(url)
        if u.scheme not in ("http","https"):
            raise NetDenied("scheme_not_allowed")
        host = u.hostname or ""
        # rate limit per host
        b = NetSandbox._buckets.setdefault(host, TokenBucket(5.0, 10))
        if not b.take():
            raise NetRateLimit("rate_limited")
        port = u.port or (443 if u.scheme=="https" else 80)
        path = u.path or "/"
        if u.query: path += "?"+u.query

        # raw tcp
        sock = socket.create_connection((host, port), timeout=timeout)
        if u.scheme=="https":
            ctx = ssl.create_default_context()
            sock = ctx.wrap_socket(sock, server_hostname=host)
        req = f"GET {path} HTTP/1.1\r\nHost: {host}\r\nUser-Agent: IMU/net-sandbox\r\nConnection: close\r\n\r\n"
        sock.sendall(req.encode())
        chunks=[]
        while True:
            data = sock.recv(8192)
            if not data: break
            chunks.append(data)
        sock.close()
        raw = b"".join(chunks)
        # strip headers
        try:
            head, body = raw.split(b"\r\n\r\n",1)
        except ValueError:
            body = raw
        return body.decode("utf-8", "replace")

    def _check_host(self, host: str):
        if self.allow_hosts and host not in self.allow_hosts:
            raise NetError(f"host_not_allowed:{host}")

    def open(self, host: str, port: int) -> str:
        self._check_host(host)
        s = socket.create_connection((host, port), timeout=self.timeout)
        s.settimeout(self.timeout)
        cid = f"{host}:{port}:{id(s)}"
        self.conns[cid] = s
        return cid

    def close(self, conn_id: str) -> None:
        s = self.conns.pop(conn_id, None)
        if s:
            try: s.close()
            except Exception: pass
