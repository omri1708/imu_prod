# imu_repo/sandbox/fs_net.py
from __future__ import annotations
import os, http.client, ssl, time
from typing import Dict, Any
from urllib.parse import urlparse
from grounded.source_policy import policy_singleton as SourcePolicy
from sandbox.net_rl import RateLimiter
from sandbox.net_class_rl import ClassRateLimiter

class SandboxViolation(Exception): ...
class QuotaExceeded(Exception): ...

class FSSandbox:
    def __init__(self, root: str, byte_quota: int = 5_000_000):
        self.root = os.path.abspath(root); self.quota = byte_quota; self.bytes = 0
        os.makedirs(self.root, exist_ok=True)
    def _resolve(self, p: str) -> str:
        ap = os.path.abspath(os.path.join(self.root, p.lstrip("/")))
        if not ap.startswith(self.root): raise SandboxViolation("path_escape")
        return ap
    def write(self, rel: str, data: bytes):
        if self.bytes + len(data) > self.quota: raise QuotaExceeded("fs_quota")
        path = self._resolve(rel); os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path,"wb") as f: f.write(data); self.bytes += len(data)
    def read(self, rel: str) -> bytes:
        path = self._resolve(rel)
        with open(path,"rb") as f: return f.read()

class NetSandbox:
    def __init__(self, max_bytes: int = 2_000_000):
        self.bytes = 0; self.max = max_bytes
        self.dom_rl = RateLimiter()
        self.cls_rl = ClassRateLimiter()
    def http_get(self, url: str, timeout_s: float = 4.0) -> Dict[str,Any]:
        if self.bytes >= self.max: raise QuotaExceeded("net_quota")
        if not (url.startswith("http://") or url.startswith("https://")): raise SandboxViolation("scheme")
        if not SourcePolicy.domain_allowed(url): raise SandboxViolation("domain_not_allowed")
        # Rate-limit כפול: דומיין + class
        if not self.dom_rl.allow(url, 4096): raise QuotaExceeded("rate_domain")
        if not self.cls_rl.allow(url, 4096): raise QuotaExceeded("rate_class")
        u = urlparse(url); host = u.hostname; port = u.port or (443 if u.scheme=="https" else 80)
        path = (u.path or "/") + (("?" + u.query) if u.query else "")
        if u.scheme == "https":
            conn = http.client.HTTPSConnection(host, port, timeout=timeout_s, context=ssl.create_default_context())
        else:
            conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
        conn.request("GET", path, headers={"Accept":"application/json"})
        r = conn.getresponse(); data = r.read()
        self.bytes += len(data)
        return {"status": r.status, "headers": dict(r.getheaders()), "body": data, "ts": time.time(), "url": url}