# imu_repo/tests/test_stage54_runtime_budget_gate.py
from __future__ import annotations
import asyncio, threading, json, time, http.server, socketserver
from runtime.async_sandbox import SandboxRuntime
from runtime.metrics import metrics
from engine.gates.runtime_budget import RuntimeBudgetGate

PORT = 8132

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"ok": True}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type","application/json")
        self.end_headers()
        self.wfile.write(body)

def run_server():
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        httpd.serve_forever()

async def produce_load():
    t = threading.Thread(target=run_server, daemon=True); t.start()
    await asyncio.sleep(0.1)
    metrics.reset()
    sbx = SandboxRuntime(allow_hosts=["127.0.0.1"], http_tps=10.0, max_sleep_ms=500)
    # 6 קריאות → counter=6; זמן קצר → p95 נמוך
    for _ in range(6):
        await sbx.http_get("127.0.0.1", PORT, "/")
    await sbx.sleep_ms(50)

def run():
    asyncio.run(produce_load())
    gate_ok = RuntimeBudgetGate(p95={"sandbox.http_get": 800, "sandbox.sleep_ms": 300},
                                counters_max={"sandbox.http_get.count": 10})
    res_ok = gate_ok.check()
    # הפחת את הספים כדי לגרום לכשל
    gate_bad = RuntimeBudgetGate(p95={"sandbox.http_get": 1}, counters_max={"sandbox.http_get.count": 2})
    res_bad = gate_bad.check()
    ok = res_ok["ok"] and (not res_bad["ok"])
    print("OK" if ok else "FAIL")
    return 0 if ok else 1

if __name__=="__main__":
    raise SystemExit(run())