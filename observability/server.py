# imu_repo/observability/server.py
from __future__ import annotations
import http.server, socketserver, json, os, time, threading
from typing import Any, Dict
from metrics.aggregate import aggregate_metrics, _iter_jsonl
from alerts.notifier import _alert_f, _metrics_f  # שימוש נתיבים קיימים

HOST="127.0.0.1"; PORT=8799

def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a, **k): pass
    def _200(self, ctype="application/json"):
        self.send_response(200); self.send_header("Content-Type", ctype); self.end_headers()

    def do_GET(self):
        try:
            if self.path.startswith("/metrics.json"):
                name = "guarded_handler"
                win  = 600
                base = aggregate_metrics(name=name, bucket="baseline", window_s=win)
                can  = aggregate_metrics(name=name, bucket="canary",   window_s=win)
                allb = aggregate_metrics(name=name, bucket=None,       window_s=win)
                self._200(); self.wfile.write(_json_bytes({"baseline":base,"canary":can,"all":allb})); return
            if self.path.startswith("/alerts.json"):
                rows=[]
                for i, a in enumerate(_iter_jsonl(_alert_f)):
                    if i>999: break
                    rows.append(a)
                self._200(); self.wfile.write(_json_bytes({"alerts": rows})); return
            if self.path == "/" or self.path.endswith(".html"):
                self._200("text/html; charset=utf-8")
                html = """<!doctype html><meta charset="utf-8">
<title>IMU Observability</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:20px}
pre{white-space:pre-wrap;border:1px solid #ddd;padding:10px}
h1,h2{margin:0.2em 0}
</style>
<h1>IMU Observability</h1>
<p>מדדים אחרונים (p95 / error-rate / throughput) לקנרית ולבייסליין, ועוד.</p>
<p><button onclick="load()">refresh</button></p>
<pre id="out">loading...</pre>
<script>
async function load(){
  const m = await fetch('/metrics.json').then(r=>r.json());
  const a = await fetch('/alerts.json').then(r=>r.json());
  const out = document.getElementById('out');
  out.textContent = JSON.stringify({metrics:m, alerts:a}, null, 2);
}
load();
</script>
"""             
                self.wfile.write(html.encode("utf-8")); return
                return
            self.send_error(404)
        except Exception as e:
            self.send_error(500, str(e))

def run(host: str=HOST, port: int=PORT) -> threading.Thread:
    httpd = socketserver.TCPServer((host, port), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return t