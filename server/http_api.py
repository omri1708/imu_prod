# server/http_api.py 
import json, base64
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
from engine.respond import GroundedResponder
from provenance.store import CASStore
from realtime.integrations import push_progress, push_timeline, start_realtime 


CAS_DIR = ".imu_cas"
KEYS_DIR = ".imu_keys"
responder = GroundedResponder(trust_threshold=0.6)
cas = CASStore(CAS_DIR, KEYS_DIR)

def _j(s, code=200):
    return (code, {"Content-Type":"application/json; charset=utf-8"}, json.dumps(s, ensure_ascii=False).encode("utf-8"))

class API(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            ln = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(ln) if ln>0 else b"{}"
            body = json.loads(raw or "{}")
        except Exception as e:
            self._send(*_j({"ok":False,"error":f"bad_json:{e}"},400)); return

        path = urlparse(self.path).path
        if path == "/api/cas/put":
            b = base64.b64decode(body.get("bytes",""))
            meta = cas.put_bytes(b, sign=True, url=body.get("url"), trust=float(body.get("trust",0.5)),
                                 not_after_days=int(body.get("not_after_days",7)))
            self._send(*_j({"ok":True,"sha256":meta.sha256,"url":meta.url,"trust":meta.trust})); return

        if path == "/api/respond":
            ctx = {"__claims__": body.get("claims", [])}
            try:
                out = responder.respond(ctx, body.get("text",""))
            except Exception as e:
                self._send(*_j({"ok":False,"error":str(e)},403)); return
            self._send(*_j(out,200)); return

        # פרסומי סטרים:
        if path == "/api/progress/update":
            # body: {"id":"build1","value": 37}
            pid = str(body["id"]); val = int(body["value"])
            start_realtime(f"progress/{pid}", {"value": val})
            self._send(*_j({"ok":True})); return

        if path == "/api/timeline/add":
            # body: {"stream":"build","event":{"type":"info","ts":1690000000,"text":"..."}}
            st = str(body.get("stream","main"))
            ev = body.get("event",{})
            start_realtime(f"timeline/{st}", {"event": ev})
            self._send(*_j({"ok":True})); return

        self._send(*_j({"ok":False,"error":"not_found"},404))

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>IMU RT</title></head>
<body>
<script>{open('ui_dsl/client_ws.js','r',encoding='utf-8').read()}</script>
<h1>IMU Real-time</h1>
<div id="app"></div>
<script>
window.IMU_WS_URL = "ws://"+location.hostname+":8766/ws";
document.getElementById('app').innerHTML = `
  <div>
    <h3>Progress</h3>
    {{}}
    <h3>Timeline</h3>
    {{}}
  </div>`;
</script>
</body></html>"""
            self._send(200, {"Content-Type":"text/html; charset=utf-8"}, html.encode("utf-8")); return
        self._send(*_j({"ok":False,"error":"not_found"},404))

    def _send(self, code, headers, data):
        self.send_response(code)
        for k,v in headers.items(): self.send_header(k,v)
        self.end_headers(); self.wfile.write(data)

def serve(host="127.0.0.1", port=8765, ws_host="127.0.0.1", ws_port=8766):
    start_ws_broker(ws_host, ws_port)
    httpd = HTTPServer((host, port), API)
    print(f"[imu] http api on http://{host}:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    serve()