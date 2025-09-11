# imu_repo/tests/test_stage54_realtime_runtime.py
from __future__ import annotations
import asyncio, threading, time, json, http.server, socketserver
from runtime.async_sandbox import SandboxRuntime, PolicyError, ThrottleExceeded
from runtime.metrics import metrics

PORT = 8131

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path=="/ok":
            body = json.dumps({"ok": True, "t": int(time.time())}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()

def run_server():
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        httpd.serve_forever()

async def main():
    t = threading.Thread(target=run_server, daemon=True); t.start()
    await asyncio.sleep(0.1)

    metrics.reset()
    sbx = SandboxRuntime(allow_hosts=["127.0.0.1","localhost"], http_tps=2.0, max_sleep_ms=300)

    # sleep תקין
    await sbx.sleep_ms(100)

    # HTTP OK
    s,h,b = await sbx.http_get("127.0.0.1", PORT, "/ok")
    ok1 = (s==200 and b.startswith(b"{"))

    # allowlist: מארח לא מותר
    err1=False
    try:
        await sbx.http_get("example.com", 80, "/")
    except PolicyError:
        err1=True

    # throttling: 2 TPS -> שלישית תיזרק
    err2=False
    await sbx.http_get("127.0.0.1", PORT, "/ok")
    try:
        await sbx.http_get("127.0.0.1", PORT, "/ok")
    except ThrottleExceeded:
        err2=True

    # sleep חורג מדיניות
    err3=False
    try:
        await sbx.sleep_ms(1000)
    except PolicyError:
        err3=True

    print("OK" if (ok1 and err1 and err2 and err3) else "FAIL")
    return 0 if (ok1 and err1 and err2 and err3) else 1

if __name__=="__main__":
    raise SystemExit(asyncio.run(main()))