# http/api.py
from http.server import BaseHTTPRequestHandler, HTTPServer
import json, threading
from typing import Dict, Any
from engine.pipeline_bindings import run_adapter
from broker.streams import Broker
from contracts.base import ResourceRequired

BROKER = Broker.singleton()

class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, obj: Dict[str, Any]):
        self.send_response(code)
        self.send_header("Content-Type","application/json"); self.end_headers()
        self.wfile.write(json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def do_POST(self):
        if self.path == "/run_adapter":
            ln = int(self.headers.get("Content-Length","0")); body = self.rfile.read(ln)
            req = json.loads(body or b"{}")
            name = req.get("name"); params = req.get("params",{})
            task_id = req.get("task_id","adp-"+name)
            BROKER.publish("timeline", {"task_id":task_id,"event":"accepted","adapter":name})
            def _work():
                try:
                    BROKER.publish("progress", {"task_id":task_id,"pct":5,"msg":"starting"})
                    res = run_adapter(name, **params)
                    BROKER.publish("progress", {"task_id":task_id,"pct":95,"msg":"finalizing"})
                    BROKER.publish("timeline", {"task_id":task_id,"event":"finished","ok":res.ok,"cid":res.provenance_cid})
                    BROKER.publish("artifact", {"task_id":task_id,"path":res.artifact_path,"cid":res.provenance_cid})
                    BROKER.publish("progress", {"task_id":task_id,"pct":100,"ok":res.ok})
                except ResourceRequired as e:
                    BROKER.publish("timeline", {"task_id":task_id,"event":"resource_required","what":e.what,"how_to":e.how_to})
                except Exception as e:
                    BROKER.publish("timeline", {"task_id":task_id,"event":"error","err":str(e)})
            threading.Thread(target=_work, daemon=True).start()
            return self._json(202, {"status":"accepted","task_id":task_id})
        return self._json(404, {"error":"not_found"})

def serve(addr: str="127.0.0.1", port: int=8088):
    HTTPServer((addr,port), Handler).serve_forever()