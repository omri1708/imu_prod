# imu_repo/adapters/http_fetch.py
from __future__ import annotations
import urllib.request, urllib.parse, ssl, socket, Request, urlopen
import ssl, time
from typing import Dict, Any, Optional, Tuple
from urllib.error import URLError, HTTPError

class HTTPError(Exception): ...


def http_fetch(url: str,
               method: str = "GET",
               headers: Optional[Dict[str,str]] = None,
               body: Optional[bytes] = None,
               timeout: float = 10.0,
               allow_hosts: Optional[list[str]] = None) -> Dict[str,Any]:
    """
    Minimal secure HTTP fetch using urllib (no external deps).
    - Optional allowlist of hostnames.
    - TLS verification on by default.
    """
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    if allow_hosts is not None and host not in allow_hosts:
        raise HTTPError(f"host_not_allowed:{host}")

    req = urllib.request.Request(url=url, method=method.upper(), data=body)
    for k,v in (headers or {}).items():
        req.add_header(k, v)

    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            data = resp.read()
            return {
                "status": resp.getcode(),
                "headers": dict(resp.headers),
                "body": data
            }
    except (urllib.error.HTTPError, urllib.error.URLError, socket.timeout) as e:
        raise HTTPError(str(e))


def http_fetch_bytes(url: str, timeout: float = 2.0) -> Tuple[bytes, Dict[str,Any]]:
    """
    מביא תוכן ו־Headers בסיסיים. ללא ספריות חיצוניות.
    מחזיר (bytes, meta_headers)
    """
    req = Request(url, headers={"User-Agent":"IMU/strict-ground"})
    ctx = ssl.create_default_context()
    try:
        with urlopen(req, timeout=timeout, context=ctx) as r:
            b = r.read()
            hdrs = {k.lower():v for k,v in r.headers.items()}
            meta = {
                "source": url,
                "kind": "http",
                "etag": hdrs.get("etag"),
                "last_modified": hdrs.get("last-modified"),
                "content_type": hdrs.get("content-type"),
                "fetched_at": time.time()
            }
            return b, meta
    except (HTTPError, URLError) as e:
        raise RuntimeError(f"http_fetch_failed:{e}")