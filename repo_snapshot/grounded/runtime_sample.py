# imu_repo/grounded/runtime_sample.py
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple
import io, json, urllib.request, urllib.parse, ipaddress, ssl

class RuntimeFetchError(Exception): ...
class RuntimePolicyError(Exception): ...

def _is_private_host(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast
    except ValueError:
        # Hostname, not IP — נחסום localhost ודומיו
        h = host.lower()
        return h in ("localhost",) or h.endswith(".local")

def _safe_parse(url: str) -> urllib.parse.ParseResult:
    pr = urllib.parse.urlparse(url)
    if pr.scheme not in ("http", "https"):
        raise RuntimePolicyError(f"scheme not allowed: {pr.scheme}")
    if not pr.netloc:
        raise RuntimePolicyError("missing host")
    host = pr.hostname or ""
    if _is_private_host(host):
        raise RuntimePolicyError(f"private/loopback host not allowed: {host}")
    return pr

def default_fetcher(url: str, *, timeout: float, max_bytes: int) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json,*/*;q=0.8","User-Agent":"imu-runtime-guard/1.0"}
    )
    # SSL: ברירת־מחדל בטוחה
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        data = resp.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise RuntimeFetchError(f"response too large (> {max_bytes} bytes)")
    return data

def _as_rows(obj: Any) -> List[Dict[str,Any]]:
    """
    תומך במבנים נפוצים:
      - [{"col":...}, ...]
      - {"items":[...]} / {"data":[...]} / {"results":[...]}
    """
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for k in ("items","data","results","rows"):
            v = obj.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    raise RuntimeFetchError("unsupported JSON shape (expect list[object] or {items|data|results|rows})")

def fetch_sample_json(
    url: str,
    *,
    timeout_s: float = 3.0,
    max_bytes: int = 1_000_000,
    sample_limit: int = 200,
    fetcher: Optional[Callable[[str], bytes]] = None
) -> List[Dict[str,Any]]:
    """
    דוגם עד sample_limit רשומות JSON ממקור חי (או fetcher מוזרק בבדיקה).
    אוכף מדיניות בטיחות בסיסית כדי למנוע SSRF/localhost וכד'.
    """
    _safe_parse(url)
    fetch = (lambda u: default_fetcher(u, timeout=timeout_s, max_bytes=max_bytes))
    if fetcher is not None:
        def fetch(u: str) -> bytes:  # type: ignore
            return fetcher(u)
    raw = fetch(url)
    try:
        obj = json.loads(raw.decode("utf-8", errors="strict"))
    except Exception as e:
        raise RuntimeFetchError(f"invalid JSON: {e}")
    rows = _as_rows(obj)
    return rows[: int(sample_limit)]

def fetch_sample_with_raw(
    url: str,
    *,
    timeout_s: float = 3.0,
    max_bytes: int = 1_000_000,
    sample_limit: int = 200,
    fetcher=None
):
    """
    מחזיר (rows, raw_bytes). אם fetcher מוזרק — משמש גם כאן.
    """
    _safe_parse(url)
    fetch = (lambda u: default_fetcher(u, timeout=timeout_s, max_bytes=max_bytes))
    if fetcher is not None:
        def fetch(u: str) -> bytes:  # type: ignore
            return fetcher(u)
    raw = fetch(url)
    try:
        obj = json.loads(raw.decode("utf-8", errors="strict"))
    except Exception as e:
        raise RuntimeFetchError(f"invalid JSON: {e}")
    rows = _as_rows(obj)
    return rows[: int(sample_limit)], raw