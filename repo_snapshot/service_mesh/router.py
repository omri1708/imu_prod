# imu_repo/service_mesh/router.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import asyncio, time
from runtime.async_sandbox import SandboxRuntime
from runtime.metrics import metrics, atimer
from service_mesh.health import BackendState, HealthChecker
from service_mesh.policy import backoff_schedule

HTTP_OK = {200:"OK",201:"Created",202:"Accepted",204:"No Content"}
HTTP_ERR = {400:"Bad Request",403:"Forbidden",404:"Not Found",429:"Too Many Requests",500:"Internal Server Error",502:"Bad Gateway",503:"Service Unavailable"}

class Router:
    """
    Router HTTP מינימלי:
      - מקבל GET/HEAD בלבד
      - בוחר backend ע"פ score(), נמנע מ-load_shed, מכבד circuit
      - Retries עם backoff בין backends
      - מודד p95/שגיאות, כותב counters
    """
    def __init__(self, routes: Dict[str, List[Dict[str,Any]]], *, port: int=8151):
        self.port = int(port)
        self.routes = routes
        self.backends: Dict[str, BackendState] = {}
        for svc, arr in routes.items():
            for cfg in arr:
                name = cfg["name"]
                if name in self.backends: continue
                self.backends[name] = BackendState(name, cfg["host"], int(cfg["port"]),
                                                   ewma_alpha=cfg.get("ewma_alpha",0.2),
                                                   max_ewma_ms=cfg.get("max_ewma_ms",1500.0),
                                                   max_inflight=cfg.get("max_inflight",64),
                                                   fail_open_s=cfg.get("fail_open_s",6.0))
        self._hc = HealthChecker(self.backends, interval_s=1.0)
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", self.port)
        asyncio.create_task(self._hc.run())

    async def stop(self) -> None:
        self._hc.stop()
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    def _match(self, path: str) -> List[BackendState]:
        # מיפוי לפי prefix route (למשל "/hello")
        best_len = -1
        chosen: List[Dict[str,Any]] = []
        for prefix, backends in self.routes.items():
            if path.startswith(prefix) and len(prefix) > best_len:
                chosen = backends; best_len = len(prefix)
        return [self.backends[c["name"]] for c in chosen]

    def _pick(self, cands: List[BackendState]) -> BackendState | None:
        # בחר backend עם score מקסימום
        if not cands: return None
        best = None; best_s = -1e18
        for b in cands:
            if b.circuit_open(): continue
            s = b.score()
            if s > best_s:
                best_s = s; best = b
        return best

    async def _forward_get(self, b: BackendState, path: str) -> Tuple[int, Dict[str,str], bytes]:
        sbx = SandboxRuntime(allow_hosts=["127.0.0.1","localhost"], http_tps=50.0, max_sleep_ms=2000)
        b.inflight += 1
        t0 = time.perf_counter()
        try:
            status, hdrs, body = await sbx.http_get(b.host, b.port, path, timeout_s=1.5)
            dt = (time.perf_counter() - t0)*1000.0
            b.record_latency(dt)
            if 200 <= status < 500:
                b.mark_ok()
            else:
                b.mark_fail()
            metrics.record_latency_ms(f"mesh.backend.latency.{b.name}", dt)
            return status, hdrs, body
        finally:
            b.inflight -= 1

    async def _serve_error(self, writer: asyncio.StreamWriter, code: int, note: str="") -> None:
        reason = HTTP_ERR.get(code, "Error")
        body = f'{{"ok":false,"error":{code},"reason":"{reason}","note":"{note}"}}'.encode("utf-8")
        head = f"HTTP/1.1 {code} {reason}\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode("utf-8")
        writer.write(head+body); await writer.drain()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        try:
            raw = await reader.readuntil(b"\r\n\r\n")
        except asyncio.IncompleteReadError:
            writer.close(); return
        try:
            top, *hdrs = raw.decode("latin1", errors="ignore").split("\r\n")
            parts = top.split()
            if len(parts)<3: return await self._serve_error(writer, 400, "bad_request_line")
            method, path, ver = parts[0], parts[1], parts[2]
            if method not in ("GET","HEAD"):
                return await self._serve_error(writer, 403, "method_not_allowed")
        except Exception:
            return await self._serve_error(writer, 400, "parse_error")

        metrics.inc("mesh.router.total", 1)

        cands = self._match(path)
        if not cands:
            metrics.inc("mesh.router.errors", 1)
            return await self._serve_error(writer, 404, "no_route")

        # נסה עד N ניסיונות על פני backends שונים + backoff
        attempts = 3
        last_status = 502; last_body=b""
        async with atimer("mesh.router.request"):
            for wait_ms in backoff_schedule(attempts=attempts, base_ms=40, max_ms=400, jitter=0.3):
                b = self._pick(cands)
                if b is None:
                    await asyncio.sleep(wait_ms/1000.0)
                    continue
                try:
                    status, hdrs, body = await self._forward_get(b, path)
                    if 200 <= status < 300:
                        reason = HTTP_OK.get(status, "OK")
                        if method == "HEAD":
                            head = f"HTTP/1.1 {status} {reason}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".encode("utf-8")
                            writer.write(head)
                        else:
                            resp = body
                            head = f"HTTP/1.1 {status} {reason}\r\nContent-Type: application/octet-stream\r\nContent-Length: {len(resp)}\r\nConnection: close\r\n\r\n".encode("utf-8")
                            writer.write(head+resp)
                        await writer.drain()
                        return
                    else:
                        last_status = status; last_body = body
                except Exception as e:
                    last_status = 502; last_body = str(e).encode("utf-8")
                # backoff לפני ניסיון נוסף
                await asyncio.sleep(wait_ms/1000.0)

        metrics.inc("mesh.router.errors", 1)
        # החזר את השגיאה האחרונה
        reason = HTTP_ERR.get(last_status, "Bad Gateway")
        body = last_body or b""
        head = f"HTTP/1.1 {last_status} {reason}\r\nContent-Type: application/octet-stream\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode("utf-8")
        writer.write(head+body); await writer.drain()