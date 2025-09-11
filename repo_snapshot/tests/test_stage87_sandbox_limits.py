# imu_repo/tests/test_stage87_sandbox_limits.py
from __future__ import annotations
import os, asyncio, time, socket, threading
from typing import Tuple

from engine.config import load_config, save_config
from engine.user_scope import user_scope
from sandbox.fs import write_text, read_text, list_tree
from sandbox.net_client import http_get

def _set_net_cfg():
    cfg = load_config()
    cfg["net"] = {
        "allow": ["localhost", "127.0.0.1"],
        "deny": [],
        "timeout_s": 2.0,
        "max_bytes": 64_000,
        "per_host_rps": 2.0,
        "burst": 2
    }
    save_config(cfg)

def _start_http_server() -> Tuple[str, int, threading.Thread]:
    # שרת HTTP מינימלי על loopback, משיב 200 עם גוף קטן
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(50)
    host, port = srv.getsockname()

    def run():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                break
            with conn:
                data = b""
                # קרא כתובת
                conn.settimeout(1.0)
                try:
                    data = conn.recv(4096)
                except Exception:
                    pass
                # השב
                body = b"hello"
                resp = (b"HTTP/1.1 200 OK\r\n"
                        b"Content-Type: text/plain\r\n"
                        b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n"
                        b"Connection: close\r\n\r\n" + body)
                try:
                    conn.sendall(resp)
                except Exception:
                    pass
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return host, port, t

def test_fs_sandbox_isolation():
    with user_scope("alice"):
        p = write_text("proj/readme.txt", "hi alice")
        assert p.endswith("alice/proj/readme.txt")
        assert read_text("proj/readme.txt") == "hi alice"
        tree = list_tree(".")
        assert "alice/proj/readme.txt".endswith(tree[0])
    # נסיון פריצה ../ אמור להיזרק
    ok = False
    try:
        with user_scope("alice"):
            write_text("../escape.txt", "nope")
    except PermissionError:
        ok = True
    assert ok, "fs_escape should raise PermissionError"

def test_net_allow_and_rate_limit():
    _set_net_cfg()
    host, port, _t = _start_http_server()
    url = f"http://{host}:{port}/ok"
    t0 = time.time()
    # נבצע 5 קריאות; ה-burst=2, rps=2 → אמור לקחת לפחות ~1.5–2.0s
    async def go():
        async def one():
            with user_scope("carol"):
                return await http_get(url)
        tasks = [asyncio.create_task(one()) for _ in range(5)]
        outs = await asyncio.gather(*tasks)
        return outs
    outs = asyncio.get_event_loop().run_until_complete(go())
    t1 = time.time()
    assert all(o.get("status") == 200 for o in outs), f"bad statuses: {outs}"
    assert (t1 - t0) >= 1.0, f"rate limit ineffective: elapsed={t1 - t0:.3f}s"

def test_net_deny():
    _set_net_cfg()
    # דומיין לא מאושר — אסור לבצע בכלל (נזרקת שגיאה לפני ניסיון חיבור)
    async def go():
        try:
            with user_scope("dave"):
                await http_get("http://example.com/")
            return False
        except PermissionError:
            return True
    ok = asyncio.get_event_loop().run_until_complete(go())
    assert ok, "net_deny should raise PermissionError"

def run():
    test_fs_sandbox_isolation()
    test_net_allow_and_rate_limit()
    test_net_deny()
    print("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())