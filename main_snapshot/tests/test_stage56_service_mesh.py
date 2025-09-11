# imu_repo/tests/test_stage56_service_mesh.py
from __future__ import annotations
import threading, time, json, http.server, socketserver, http.client
import random
from service_mesh.router import Router
from runtime.metrics import metrics
from engine.gates.slo_gate import SLOGate

PORT_FAST = 8152
PORT_FLAKY = 8153
PORT_PROXY = 8151

# --- Backends ---

class FastHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path=="/health":
            body = b"ok"
            self.send_response(200); self.send_header("Content-Type","text/plain")
            self.end_headers(); self.wfile.write(body); return
        elif self.path.startswith("/hello"):
            body = json.dumps({"ok": True, "from":"fast","t": int(time.time())}).encode("utf-8")
            self.send_response(200); self.send_header("Content-Type","application/json")
            self.end_headers(); self.wfile.write(body); return
        else:
            self.send_response(404); self.end_headers()

class FlakyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path=="/health":
            # לפעמים בריא, לפעמים לא
            if random.random() < 0.6:
                self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
            else:
                self.send_response(503); self.end_headers()
            return
        elif self.path.startswith("/hello"):
            # 40% כישלון / האטה
            if random.random() < 0.4:
                time.sleep(0.2)
                self.send_response(500); self.end_headers(); self.wfile.write(b"boom")
            else:
                time.sleep(0.03)
                body = json.dumps({"ok": True, "from":"flaky"}).encode("utf-8")
                self.send_response(200); self.send_header("Content-Type","application/json")
                self.end_headers(); self.wfile.write(body)
            return
        else:
            self.send_response(404); self.end_headers()

def run_server(port: int, handler):
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        httpd.serve_forever()

# --- Test ---

def http_get(port: int, path: str="/hello"):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
    c.request("GET", path)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data

def run():
    random.seed(1234)

    # הפעל שרתי backend
    t_fast  = threading.Thread(target=run_server, args=(PORT_FAST, FastHandler), daemon=True); t_fast.start()
    t_flaky = threading.Thread(target=run_server, args=(PORT_FLAKY, FlakyHandler), daemon=True); t_flaky.start()
    time.sleep(0.1)

    # Router
    routes = {
        "/hello": [
            {"name":"fast","host":"127.0.0.1","port":PORT_FAST, "max_inflight":64, "max_ewma_ms":800.0},
            {"name":"flaky","host":"127.0.0.1","port":PORT_FLAKY, "max_inflight":64, "max_ewma_ms":800.0},
        ]
    }
    r = Router(routes, port=PORT_PROXY)
    loop_thread = threading.Thread(target=asyncio_run, args=(r,), daemon=True); loop_thread.start()
    time.sleep(0.2)  # תן ל-health להתחיל

    metrics.reset()

    # שלח 30 בקשות דרך הפרוקסי
    ok=0; fail=0
    for _ in range(30):
        st, data = http_get(PORT_PROXY, "/hello")
        if st==200: ok+=1
        else: fail+=1
        time.sleep(0.01)

    # עצור
    asyncio_signal_stop(r)
    time.sleep(0.2)

    # תנאי הצלחה: רוב הבקשות מצליחות (router בוחר fast/עושה retry),
    # ושער ה-SLO עובר עם סף סביר.
    total = ok + fail
    errors = fail
    erate = errors/total if total>0 else 0.0

    gate = SLOGate(p95_ms={"mesh.router.request": 600},
                   error_rate_max=0.25,  # 25% מותר בטסט עם backend בעייתי
                   min_requests=10)
    res = gate.check()
    passed = (ok >= 20) and res["ok"]

    print("OK" if passed else f"FAIL (ok={ok}, fail={fail}, res={res})")
    return 0 if passed else 1

# --- asyncio helpers ---

import asyncio
async def _run_router(r: Router):
    await r.start()
    try:
        while True:
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        await r.stop()
        raise

def asyncio_run(r: Router):
    asyncio.run(_run_router(r))

def asyncio_signal_stop(r: Router):
    # שולח ביטול דרך יצירת לולאה זמנית שמאתרת את ה-task הראשי ומבטלת
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # אין לולאה נוכחית (רץ בחוט אחר) – ניצור משימה שמבטלת באמצעות asyncio.run() קצר
        async def _cancel():
            for task in asyncio.all_tasks():
                if task.get_coro().__name__ == "_run_router":
                    task.cancel()
        try:
            asyncio.run(_cancel())
        except Exception:
            pass
    else:
        for task in asyncio.all_tasks(loop):
            if task.get_coro().__name__ == "_run_router":
                task.cancel()

if __name__=="__main__":
    raise SystemExit(run())