# http/api.py
from http.server import BaseHTTPRequestHandler, HTTPServer
import json, threading
from typing import Dict, Any
from engine.pipeline_bindings import run_adapter
from broker.streams import Broker
from contracts.base import ResourceRequired
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Any
from policy.policy_engine import PolicyStore, UserPolicy
from perf.measure import PerfRegistry
from provenance.store import CAStore
from engine.enforcement import Enforcement, EvidenceError
from engine.synthesis_pipeline import SynthesisPipeline


BROKER = Broker.singleton()
POLICIES = PolicyStore()
PERF = PerfRegistry()
CASTORE = CAStore(root="./ca_store", secret_key=b"dev-secret")
ENFORCE = Enforcement(POLICIES, CASTORE, PERF)


def demo_steps():
    # דוגמאות “עובדות”: בכל שלב מספקים evidence_bytes (למשל תוצרים/דוחות JSON)
    def plan():
        ev = json.dumps({"plan":"build mobile app w/ unity"}, ensure_ascii=False).encode()
        return {"evidence_bytes": ev, "evidence_source":"planner"}
    def generate():
        ev = json.dumps({"files":["Assets/Main.cs","Scenes/Menu.unity"]}, ensure_ascii=False).encode()
        return {"evidence_bytes": ev}
    def test():
        ev = json.dumps({"tests":10,"passed":10}, ensure_ascii=False).encode()
        return {"evidence_bytes": ev}
    def verify():
        ev = json.dumps({"lint":"ok","licenses":"ok"}, ensure_ascii=False).encode()
        return {"evidence_bytes": ev}
    def package():
        ev = json.dumps({"bundle":"unity-bundle-001"}, ensure_ascii=False).encode()
        return {"evidence_bytes": ev}
    return {"plan":plan,"generate":generate,"test":test,"verify":verify,"package":package}

class Handler(BaseHTTPRequestHandler):
    def _send(self, code:int, payload:Dict[str,Any]):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Content-Length",str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path == "/run_pipeline":
            sz = int(self.headers.get("Content-Length","0") or "0")
            _ = self.rfile.read(sz)  # אפשר לעבד spec אמיתי מהלקוח
            pipe = SynthesisPipeline(PERF, ENFORCE, CASTORE)
            try:
                res = pipe.run(user_id="demo", steps=demo_steps())
                self._send(200, {"ok":True, "result":res, "perf":PERF.summary()})
            except (EvidenceError,) as ee:
                self._send(400, {"ok":False, "error":"evidence_error", "detail":str(ee)})
            except Exception as e:
                self._send(500, {"ok":False, "error":"server_error", "detail":str(e)})
        else:
            self._send(404, {"ok":False, "error":"not_found"})

def serve(host="127.0.0.1", port=8081):
    httpd = HTTPServer((host, port), Handler)
    httpd.serve_forever()

if __name__ == "__main__":
    serve()