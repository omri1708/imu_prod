# imu_repo/tests/test_stage50_pipeline_with_api_gate.py
from __future__ import annotations
import threading, time, json, http.server, socketserver
from synth.specs import BuildSpec, Contract
from engine.synthesis_pipeline import run_pipeline
from grounded.source_policy import policy_singleton as SourcePolicy

PORT = 8124

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/ver/hello"):
            body = {"service":"hello","version":"1.0.0","updated_at": int(time.time())}
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

def _schema():
    return {
        "type":"object",
        "properties":{
            "tests":{"type":"object"},
            "perf":{"type":"object","properties":{"p95_ms":{"type":"number","maximum":1500}}},
            "ui":{"type":"object","properties":{"score":{"type":"number","minimum":65}}}
        },
        "required":["tests","perf","ui"]
    }

def run():
    t = threading.Thread(target=run_server, daemon=True); t.start()
    time.sleep(0.2)
    SourcePolicy.set_allowlist(["127.0.0.1","localhost"])

    spec = BuildSpec(
        name="stage50_api_gate",
        kind="web_service",
        language_pref=["python"],
        ports=[19696],
        endpoints={"/hello":"hello_json","/ui":"static_ui"},
        contracts=[Contract(name="svc", schema=_schema())],
        evidence_requirements=["service_tests","perf_summary","ui_accessibility","official_api"]
    )
    setattr(spec, "extras", {
        "official_api_checks":[
            {
                "name":"hello_version",
                "url": f"http://127.0.0.1:{PORT}/ver/hello",
                "schema":{
                    "type":"object",
                    "properties":{"service":{"type":"string"},"version":{"type":"string"},"updated_at":{"type":"number"}},
                    "required":["service","version","updated_at"]
                },
                "claim_path":"version",
                "expected":"1.0.0"
            }
        ]
    })

    s = run_pipeline(spec, user_id="u50")
    ok = s["rollout"]["approved"] and "official_api" in s["evidence"] and s["evidence"]["official_api"]
    print("OK" if ok else "FAIL")
    return 0 if ok else 1

if __name__=="__main__":
    raise SystemExit(run())