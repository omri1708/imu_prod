# imu_repo/sandbox/net_client.py
from __future__ import annotations
import asyncio, urllib.request, urllib.parse, socket, ssl, time
from typing import Dict, Any, Optional, Tuple, List
from engine.config import load_config
from engine.policy_ctx import get_user
from sandbox.limits import RateLimiter
from grounded.claims import current

# Rate limiter גלובלי (ניתן לכייל דרך config)
_RL: Optional[RateLimiter] = None

def _cfg():
    cfg = load_config()
    net = dict(cfg.get("net", {}))
    # ברירות מחדל שמרניות
    net.setdefault("allow", ["localhost", "127.0.0.1"])
    net.setdefault("deny", [])
    net.setdefault("timeout_s", 5.0)
    net.setdefault("max_bytes", 512_000)  # 0.5MB
    net.setdefault("per_host_rps", 2.0)
    net.setdefault("burst", 2)
    return net

def _init_rl():
    global _RL
    net = _cfg()
    _RL = RateLimiter(rate_per_sec=float(net["per_host_rps"]), burst=int(net["burst"]))

def _host_port(url: str) -> Tuple[str, int]:
    pr = urllib.parse.urlparse(url)
    host = pr.hostname or ""
    port = pr.port or (443 if pr.scheme == "https" else 80)
    return host, port

def _enforce_policy(url: str) -> None:
    net = _cfg()
    host, _ = _host_port(url)
    host_l = (host or "").lower()
    # deny גובר על allow
    if any(host_l == d.lower() or host_l.endswith("." + d.lower()) for d in net.get("deny", [])):
        raise PermissionError(f"net_deny: {host}")
    if not any(host_l == a.lower() or host_l.endswith("." + a.lower()) for a in net.get("allow", [])):
        raise PermissionError(f"net_not_allowed: {host}")

async def http_request(method: str, url: str, *, headers: Optional[Dict[str,str]] = None, body: Optional[bytes] = None) -> Dict[str, Any]:
    """
    קריאה אסינכרונית (דרך thread) עם מגבלות:
      - Allow/Deny של דומיינים
      - Rate limit פר־משתמש ופר־Host
      - timeout וגודל מרבי
    הראיות נרשמות (http_request / http_response)
    """
    net = _cfg()
    timeout_s = float(net["timeout_s"])
    max_bytes = int(net["max_bytes"])
    uid = get_user() or "anon"
    if _RL is None:
        _init_rl()

    _enforce_policy(url)
    host, _ = _host_port(url)
    # rate-limit
    await _RL.acquire(uid, host, amount=1.0)

    # request בסביבת thread כדי לא לחסום event loop
    def _do() -> Dict[str, Any]:
        req = urllib.request.Request(url=url, method=method.upper(), headers=headers or {})
        ctx = ssl.create_default_context()
        start = time.time()
        try:
            with urllib.request.urlopen(req, data=body, timeout=timeout_s, context=ctx) as resp:
                status = int(resp.status)
                hdrs = {k.lower(): v for k,v in resp.getheaders()}
                # קריאה מדורגת עד max_bytes
                buf = bytearray()
                chunk = 64 * 1024
                while True:
                    if len(buf) >= max_bytes:
                        break
                    part = resp.read(min(chunk, max_bytes - len(buf)))
                    if not part:
                        break
                    buf.extend(part)
                took = time.time() - start
                return {"status": status, "headers": hdrs, "body": bytes(buf), "took_s": took}
        except Exception as e:
            return {"error": str(e), "status": 0, "headers": {}, "body": b"", "took_s": time.time() - start}

    # Evidence: לפני
    current().add_evidence("http_request", {
        "source_url": url,
        "trust": 0.9,
        "ttl_s": 600,
        "payload": {"method": method.upper()}
    })
    out = await asyncio.to_thread(_do)
    # Evidence: אחרי
    ev_payload = {"status": out.get("status", 0), "bytes": len(out.get("body", b"")), "host": host}
    current().add_evidence("http_response", {
        "source_url": url,
        "trust": 0.9 if out.get("status", 0) else 0.6,
        "ttl_s": 600,
        "payload": ev_payload
    })
    return out

async def http_get(url: str, *, headers: Optional[Dict[str,str]] = None) -> Dict[str, Any]:
    return await http_request("GET", url, headers=headers, body=None)

async def http_post(url: str, *, headers: Optional[Dict[str,str]] = None, data: Optional[bytes] = None) -> Dict[str, Any]:
    return await http_request("POST", url, headers=headers, body=data or b"")