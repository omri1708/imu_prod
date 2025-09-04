# imu_repo/synth/generate.py
from __future__ import annotations
import os, textwrap
from typing import Dict, Any
from synth.specs import BuildSpec

PY_SERVER = r"""
import http.server, socketserver, json, threading, time, os, sqlite3
PORT = int(%PORT%)
ROOT = os.path.dirname(__file__)
DB   = os.path.join(ROOT, "app.db")

def _ensure_db():
    conn = sqlite3.connect(DB)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT)")
        conn.commit()
    finally:
        conn.close()

class H(http.server.SimpleHTTPRequestHandler):
    def _send(self, code, body=b"", ct="text/plain"):
        self.send_response(code); self.send_header("Content-Type", ct); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, b"OK"); return
        %ENDPOINTS%
        # static files under ROOT/ui
        p = os.path.join(ROOT, "ui", self.path.lstrip("/"))
        if os.path.isdir(p): p = os.path.join(p, "index.html")
        if os.path.exists(p):
            ct = "text/plain"
            if p.endswith(".html"): ct="text/html; charset=utf-8"
            if p.endswith(".js"): ct="application/javascript"
            with open(p,"rb") as fh: self._send(200, fh.read(), ct); return
        self._send(404, b"not found")

    def do_POST(self):
        if self.path == "/kv":
            ln = int(self.headers.get("Content-Length","0") or "0")
            raw = self.rfile.read(ln)
            try:
                obj = json.loads(raw.decode("utf-8"))
                k = str(obj.get("k","")).strip(); v = str(obj.get("v",""))
                if not k: self._send(400,b"missing k"); return
                conn = sqlite3.connect(DB)
                try:
                    conn.execute("INSERT OR REPLACE INTO kv(k,v) VALUES(?,?)",(k,v)); conn.commit()
                finally:
                    conn.close()
                self._send(200, f"OK {k}={v}".encode()); return
            except Exception as e:
                self._send(400, f"bad json:{e}".encode()); return
        self._send(404,b"not found")

_ensure_db()
with socketserver.TCPServer(("127.0.0.1", PORT), H) as httpd:
    print("SERVING", PORT, flush=True)
    httpd.serve_forever()
"""

def _python_endpoints(endpoints: dict) -> str:
    out=[]
    for path, behavior in endpoints.items():
        if behavior=="hello_json":
            out.append(textwrap.dedent(f"""
                if self.path == "{path}":
                    self._send(200, json.dumps({{"message":"hello","path":"{path}"}}).encode(), "application/json"); return
            """).strip("\n"))
        elif behavior=="echo_time":
            out.append(textwrap.dedent(f"""
                if self.path == "{path}":
                    self._send(200, str(time.time()).encode()); return
            """).strip("\n"))
        elif behavior.startswith("static_file:"):
            rel = behavior.split(":",1)[1]
            out.append(textwrap.dedent(f"""
                if self.path == "{path}":
                    p = os.path.join(ROOT, "{rel}")
                    if os.path.exists(p):
                        ct="text/plain"
                        if p.endswith(".html"): ct="text/html; charset=utf-8"
                        if p.endswith(".js"): ct="application/javascript"
                        with open(p,"rb") as fh: self._send(200, fh.read(), ct)
                    else:
                        self._send(404, b"missing")
                    return
            """).strip("\n"))
        else:
            out.append(textwrap.dedent(f"""
                if self.path == "{path}":
                    self._send(200, b"{behavior.encode('utf-8').decode('utf-8')}"); return
            """).strip("\n"))
    return "\n        ".join(out)

def generate_sources(spec: BuildSpec, out_dir: str) -> Dict[str,Any]:
    os.makedirs(out_dir, exist_ok=True)
    lang = "python"
    for l in spec.language_pref:
        if l in ("python","node","go","rust"): lang = l; break

    # UI סטטי
    from ui.static_pack import write_basic_ui
    ui_index = write_basic_ui(out_dir)
    spec.endpoints.setdefault("/ui", f"static_file:{os.path.relpath(ui_index, out_dir)}")

    if lang=="python":
        code = PY_SERVER.replace("%PORT%", str(spec.ports[0] if spec.ports else 18080))
        code = code.replace("%ENDPOINTS%", _python_endpoints(spec.endpoints))
        path = os.path.join(out_dir, "server.py")
        with open(path,"w",encoding="utf-8") as f: f.write(code)
        return {"language":"python","entry":path}
    else:
        # future: node/go/rust generators – כאן נשאר בפייתון לצורך תאימות הרצה מיידית
        code = "// Alternative generator not enabled in stage 38.\n"
        path = os.path.join(out_dir, "server.mjs")
        with open(path,"w",encoding="utf-8") as f: f.write(code)
        return {"language":lang,"entry":path}