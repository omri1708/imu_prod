# security/sandbox.py
# -*- coding: utf-8 -*-
import os, socket, pathlib
from typing import Iterable
from contracts.errors import SandboxDenied

class FileSandbox:
    def __init__(self, root: str, allow_write: bool = True):
        self.root = os.path.abspath(root)
        self.allow_write = allow_write
        os.makedirs(self.root, exist_ok=True)

    def _resolve(self, p: str) -> str:
        ap = os.path.abspath(os.path.join(self.root, p.lstrip("/")))
        if not ap.startswith(self.root + os.sep) and ap != self.root:
            raise SandboxDenied("fs", p)
        return ap

    def read(self, p: str) -> bytes:
        ap = self._resolve(p)
        with open(ap, "rb") as f: return f.read()

    def write(self, p: str, data: bytes):
        if not self.allow_write: raise SandboxDenied("fs_write_disabled", p)
        ap = self._resolve(p)
        pathlib.Path(os.path.dirname(ap)).mkdir(parents=True, exist_ok=True)
        with open(ap, "wb") as f: f.write(data)

class NetSandbox:
    def __init__(self, allow_hosts: Iterable[str] = ()):
        self.allow = set(allow_hosts)

    def check_host(self, host: str):
        if not self.allow: return
        if host not in self.allow and not any(host.endswith(suffix) for suffix in self.allow):
            raise SandboxDenied("net", host)

    def connect(self, host: str, port: int, timeout: float = 5.0):
        self.check_host(host)
        s = socket.create_connection((host, port), timeout=timeout)