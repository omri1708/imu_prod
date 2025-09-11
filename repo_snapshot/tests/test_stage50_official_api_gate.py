# imu_repo/tests/test_stage50_official_api_gate.py
from __future__ import annotations
import threading, time, json, http.server, socketserver
from grounded.api_gate import OfficialAPIGate
from grounded.source_policy import policy_singleton as SourcePolicy

PORT = 8123

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/orgs/acme"):
            body = {"org":"acme","version":"3.7.5","updated_at": int(time.time())}
            b = json.dumps(body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.send_header("Last-Modified", self.date_time_string(time.time()))
            self.end_headers()
            self.wfile.write(b)
        else:
            self.send_response(404); self.end_headers()

def run_server():
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        httpd.serve_forever()

def run():
    # שרת רץ ברקע
    t = threading.Thread(target=run_server, daemon=True); t.start()
    time.sleep(0.2)

    # מתירים localhost ב-Allowlist
    SourcePolicy.set_allowlist(["127.0.0.1","localhost"])

    schema = {
        "type":"object",
        "properties":{
            "org":{"type":"string"},
            "version":{"type":"string"},
            "updated_at":{"type":"number"}
        },
        "required":["org","version","updated_at"]
    }
    gate = OfficialAPIGate(ttl_s=10*60)
    res = gate.verify(
        name="acme_version",
        url=f"http://127.0.0.1:{PORT}/orgs/acme",
        json_schema=schema,
        claim_path="version",
        expected="3.7.5",
        user_id="gate_tester",
        obj="acme_service",
        tags=["unit"]
    )
    ok = bool(res.get("ok"))
    print("OK" if ok else "FAIL")
    return 0 if ok else 1

if __name__=="__main__":
    raise SystemExit(run())