# engine/http_api.py (חיבור /run_adapter + SSE ל-UI)
# -*- coding: utf-8 -*-
from __future__ import annotations
import json, time, threading, queue, os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from typing import Callable, Dict, Any
from broker.bus import EventBus
from adapters.adapter_runner import run_adapter
from policy.user_policy import DEFAULT_POLICY
from provenance.store import ProvenanceStore, ResourceRequired

from adapters.registry import ADAPTERS
from broker.prior_broker import GLOBAL_BROKER
from policy.user_policy import must_be_grounded, per_topic_limits
from grounded.http_verifier import HttpVerifier, ExternalVerifyError

BUS = EventBus()
PROV = ProvenanceStore()

class APIServer(BaseHTTPRequestHandler):

    def _json(self, code:int, obj:Any):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/events/"):
            run_id = self.path.split("/")[-1]
            self.send_response(200)
            self.send_header("Content-Type","text/event-stream")
            self.send_header("Cache-Control","no-cache")
            self.send_header("Connection","keep-alive")
            self.end_headers()
            sub = GLOBAL_BROKER.subscribe(topic=f"run.{run_id}")
            try:
                for evt in sub:
                    msg = f"event:{evt['type']}\n"
                    data = json.dumps(evt, ensure_ascii=False)
                    self.wfile.write(msg.encode("utf-8"))
                    self.wfile.write(b"data:")
                    self.wfile.write(data.encode("utf-8"))
                    self.wfile.write(b"\n\n")
                    self.wfile.flush()
            except ConnectionResetError:
                pass
            return
        return self._json(404, {"error":"not_found"})

    def do_POST(self):
        if self.path == "/run_adapter":
            length = int(self.headers.get("Content-Length","0"))
            raw = self.rfile.read(length)
            req = json.loads(raw or b"{}")
            if not isinstance(req, dict):
                raise TypeError("expected top-level JSON object")
            user_id = req.get("user_id", "anon")
            user_id = req.get("user_id","anon")
            adapter = req["adapter"]
            run_id = req.get("run_id") or f"run_{int(time.time()*1000)}"
            args = req.get("args",{})
            # אם נדרש Grounding קשיח—נבדוק שהבקשה כוללת claims ולפחות אחת מאומתת:
            if must_be_grounded(user_id):
                claims = req.get("claims") or []
                if not claims:
                    return self._json(400, {"error":"grounding_required","detail":"claims missing"})
                ok_digests = []
                for claim in claims:
                    try:
                        res = HttpVerifier.verify_claim(user_id, claim)
                        ok_digests.append(res["digest"])
                    except ExternalVerifyError as e:
                        return self._json(400, {"error":"claim_verification_failed", "detail": str(e)})
                args["_evidence_digests"] = ok_digests

            # פרסום אירוע התחלה:
            GLOBAL_BROKER.publish(f"run.{run_id}", {"type":"start","ts":time.time(),"adapter":adapter})
            try:
                out = ADAPTERS.run(adapter, run_id=run_id, args=args, user_id=user_id)
                GLOBAL_BROKER.publish(f"run.{run_id}", {"type":"finished","ts":time.time(),"result":out})
                return self._json(200, {"ok":True, "run_id":run_id, "result":out})
            except ResourceRequired as rr:
                GLOBAL_BROKER.publish(f"run.{run_id}", {"type":"blocked","ts":time.time(),"need":rr.requirements})
                return self._json(428, {"error":"resource_required", "need": rr.requirements, "run_id":run_id})
            except Exception as ex:
                GLOBAL_BROKER.publish(f"run.{run_id}", {"type":"error","ts":time.time(),"error":str(ex)})
                return self._json(500, {"error":"adapter_failed", "detail":str(ex), "run_id":run_id})

def serve(host="127.0.0.1", port=8088):
    httpd = HTTPServer((host,port), APIServer)
    print(f"[api] listening on http://{host}:{port}")
    httpd.serve_forever()

class SSEClients:
    def __init__(self): self._clients=set(); self._lock=threading.RLock()
    def add(self, wfile): 
        with self._lock: self._clients.add(wfile)
    def discard(self, wfile):
        with self._lock:
            if wfile in self._clients: self._clients.remove(wfile)
    def broadcast(self, event:Dict[str,Any]):
        data = "data: "+json.dumps(event, ensure_ascii=False)+"\n\n"
        dead=[]
        with self._lock:
            for c in list(self._clients):
                try: c.write(data.encode("utf-8")); c.flush()
                except Exception: dead.append(c)
            for d in dead: 
                try: d.close()
                except: pass
                self._clients.discard(d)

SSE = SSEClients()

def emit(topic:str, payload:Dict[str,Any]):
    evt={"topic":topic, "ts":int(time.time()*1000), "payload":payload}
    BUS.publish(topic, evt)
    SSE.broadcast(evt)

class Handler(BaseHTTPRequestHandler):
    def _json(self, code:int, obj:Dict[str,Any]):
        self.send_response(code)
        self.send_header("Content-Type","application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(obj,ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        if self.path.startswith("/events"):
            self.send_response(200)
            self.send_header("Content-Type","text/event-stream")
            self.send_header("Cache-Control","no-cache")
            self.send_header("Connection","keep-alive")
            self.end_headers()
            SSE.add(self.wfile)
            try:
                while True: time.sleep(60)  # החזקה פתוחה
            except Exception:
                SSE.discard(self.wfile)
            return
        if self.path.startswith("/healthz"):
            return self._json(200, {"ok":True})
        return self._json(404, {"error":"not_found"})

    def do_POST(self):
        if self.path.startswith("/run_adapter"):
            length=int(self.headers.get("Content-Length","0"))
            body=json.loads(self.rfile.read(length) or b"{}")
            user=body.get("user_id","default")
            adapter=body["adapter"]
            args=body.get("args",{})
            # הרצה עם שידור timeline:
            emit("timeline", {"phase":"start","adapter":adapter,"user":user})
            try:
                result = run_adapter(adapter, args, policy=DEFAULT_POLICY, emit=emit, prov=PROV)
                emit("timeline", {"phase":"done","adapter":adapter,"user":user,"result":result})
                return self._json(200, {"ok":True,"result":result})
            except ResourceRequired as rr:
                emit("timeline", {"phase":"blocked","adapter":adapter,"user":user,"need":rr.what})
                return self._json(428, {"ok":False,"need":rr.what,"how":rr.how})
            except Exception as e:
                emit("timeline", {"phase":"error","adapter":adapter,"user":user,"error":str(e)})
                return self._json(500, {"ok":False,"error":str(e)})
        return self._json(404, {"error":"not_found"})


if __name__=="__main__":
    serve()