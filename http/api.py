# http/api.py
import  os
from http.server import BaseHTTPRequestHandler, HTTPServer
import json, threading
from typing import Dict, Any

from engine.pipeline_bindings import run_adapter
from broker.streams import Broker
from contracts.base import ResourceRequired

from policy.policy_engine import PolicyStore, UserPolicy
from perf.measure import PerfRegistry
from provenance.store import CAStore
from engine.enforcement import Enforcement, EvidenceError
from engine.synthesis_pipeline import SynthesisPipeline

from provenance.store import CAS
from provenance.audit import AuditLog
from adapters.android.build import run_android_build
from adapters.ios.build import run_ios_build
from adapters.unity.cli import run_unity_cli
from adapters.k8s.deploy import run_k8s_deploy
from adapters.cuda.runner import run_cuda_job
from engine.adapter_router import new_broker


BROKER = Broker.singleton()
POLICIES = PolicyStore()
PERF = PerfRegistry()
CASTORE = CAStore(root="./ca_store", secret_key=b"dev-secret")
ENFORCE = Enforcement(POLICIES, CASTORE, PERF)
ADAPTERS = {
    "android_build": run_android_build,
    "ios_build": run_ios_build,
    "unity_cli": run_unity_cli,
    "k8s_deploy": run_k8s_deploy,
    "cuda_job": run_cuda_job,
}

class State:
    cas: CAS = None
    audit: AuditLog = None
    broker = None


def setup_state(root: str, secret: bytes, user_id: str="user"):
    State.cas = CAS(os.path.join(root, "cas"), secret)
    State.audit = AuditLog(os.path.join(root, "audit","log.jsonl"), secret)
    State.broker = new_broker(user_id, State.cas, State.audit)

class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, payload: Dict[str,Any]):
        b = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        if self.path != "/run_adapter":
            self._json(404, {"error":"not_found"})
            return
        raw = self.rfile.read(int(self.headers.get("Content-Length","0") or "0"))
        try:
            req = json.loads(raw or b"{}")
        except Exception:
            self._json(400, {"error":"bad_json"})
            return
        name = req.get("name")
        cfg = req.get("config") or {}
        if name not in ADAPTERS:
            self._json(400, {"error":"unknown_adapter"})
            return
        # signal start → UI
        State.broker.submit("timeline", {"message": f"start {name}"})
        try:
            res = ADAPTERS[name](cfg, State.audit)
            # success event
            State.broker.submit("progress", {"progress": 100, "name": name})
            State.broker.submit("timeline", {"message": f"done {name}"})
            self._json(200, {"ok": True, "result": res})
        except Exception as e:
            State.audit.append("http","adapter_error",{"name":name,"err":str(e)})
            State.broker.submit("timeline", {"message": f"error {name}: {e}"})
            self._json(500, {"ok": False, "error": str(e)})


def run_http(root: str, secret: bytes, host="127.0.0.1", port=8787):
    setup_state(root, secret)
    httpd = HTTPServer((host, port), Handler)
    print(f"IMU HTTP listening at http://{host}:{port}")
    httpd.serve_forever()


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


def serve(host="127.0.0.1", port=8081):
    httpd = HTTPServer((host, port), Handler)
    httpd.serve_forever()


if __name__ == "__main__":
    serve()