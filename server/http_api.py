import json, threading, time, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any
from broker.stream import broker
from governance.user_policy import get_user_policy
from engine.pipeline_events import run_pipeline_spec
from audit.log import AppendOnlyAudit

AUDIT = AppendOnlyAudit("var/audit/http.jsonl")

STATIC_ROOT = os.path.abspath("ui_dsl")

class Handler(BaseHTTPRequestHandler):
    server_version = "IMU/1.0"

    def _json(self, code: int, obj: Dict[str, Any]):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _static(self, rel: str):
        p = os.path.abspath(os.path.join(STATIC_ROOT, rel))
        if not p.startswith(STATIC_ROOT) or not os.path.exists(p):
            self.send_error(404); return
        ct = "text/plain"
        if p.endswith(".js"): ct = "application/javascript; charset=utf-8"
        if p.endswith(".html"): ct = "text/html; charset=utf-8"
        with open(p, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/events":
            qs = parse_qs(u.query)
            topic = qs.get("topic", ["events"])[0]
            sub = broker.subscribe(topic, max_queue=2000)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            for chunk in broker.sse_iter(sub):
                try:
                    self.wfile.write(chunk); self.wfile.flush()
                except Exception:
                    break
            return
        if u.path.startswith("/static/"):
            rel = u.path[len("/static/"):]
            return self._static(rel)
        if u.path == "/healthz":
            return self._json(200, {"ok": True})
        return self._json(404, {"error": "not_found"})

    def do_POST(self):
        u = urlparse(self.path)
        cl = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(cl) if cl > 0 else b"{}"
        try:
            req = json.loads(raw.decode("utf-8"))
        except Exception:
            return self._json(400, {"error": "bad_json"})
        if u.path == "/v1/pipeline/run":
            user = req.get("user") or "anon"
            spec_text = req.get("spec") or "{}"
            policy, ev_index = get_user_policy(user)
            AUDIT.append({"kind":"pipeline_run_req","user":user})
            try:
                run_id = run_pipeline_spec(user=user, spec_text=spec_text, policy=policy, ev_index=ev_index)
                return self._json(200, {"ok": True, "run_id": run_id})
            except Exception as e:
                AUDIT.append({"kind":"pipeline_run_err","user":user,"err":str(e)})
                return self._json(500, {"ok": False, "error": str(e)})
        return self._json(404, {"error": "not_found"})

def serve_http(host: str = "127.0.0.1", port: int = 8080):
    srv = ThreadingHTTPServer((host, port), Handler)
    t = threading.Thread(target=srv.serve_forever, name="http", daemon=True)
    t.start()
    return srv

if __name__ == "__main__":
    print("IMU HTTP listening on :8080")
    serve_http("0.0.0.0", 8080)
    while True: time.sleep(3600)