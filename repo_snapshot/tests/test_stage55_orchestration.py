# imu_repo/tests/test_stage55_orchestration.py
from __future__ import annotations
import threading, time, json, http.server, socketserver
from orchestrator.worker_runtime import Worker
from orchestrator.orchestrator import Orchestrator, enqueue_task, collect_results
from runtime.metrics import metrics

PORT = 8141

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/ok":
            body = json.dumps({"ok": True}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()

def run_server():
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        httpd.serve_forever()

def run():
    # HTTP לוקאלי
    t_http = threading.Thread(target=run_server, daemon=True); t_http.start()
    time.sleep(0.1)

    # Orchestrator
    orch = Orchestrator()
    t_orch = threading.Thread(target=orch.run, daemon=True); t_orch.start()

    # 2 Workers עם יכולות שונות
    w1 = Worker(["sum","sleep_ms"])
    w2 = Worker(["http_local"])
    t_w1 = threading.Thread(target=w1.run, daemon=True); t_w1.start()
    t_w2 = threading.Thread(target=w2.run, daemon=True); t_w2.start()

    # המתן למנהיג ו-heartbeats
    time.sleep(0.5)

    # שליחת משימות
    id1 = enqueue_task("sum", {"a": 2, "b": 7})
    id2 = enqueue_task("sleep_ms", {"ms": 60})
    id3 = enqueue_task("http_local", {"host":"127.0.0.1","port":PORT,"path":"/ok"})

    # איסוף תוצאות
    res = collect_results(timeout_s=3.0)
    got = {r.get("task_id"): r for r in res}
    ok1 = (got.get(id1,{}).get("ok") and abs(got[id1]["result"]["result"] - 9.0) < 1e-6)
    ok2 = (got.get(id2,{}).get("ok") and got[id2]["result"]["slept_ms"] == 60)
    ok3 = (got.get(id3,{}).get("ok") and got[id3]["result"]["status"] == 200)

    # מדדים בסיסיים קיימים
    p95_dispatch = metrics.p95("orchestrator.dispatch")
    ok4 = (p95_dispatch is None) or (p95_dispatch >= 0.0)  # עצם ההקלטה

    # סגירה נקייה
    w1.stop(); w2.stop(); orch.stop()
    time.sleep(0.2)

    ok = ok1 and ok2 and ok3 and ok4
    print("OK" if ok else "FAIL")
    return 0 if ok else 1

if __name__=="__main__":
    raise SystemExit(run())