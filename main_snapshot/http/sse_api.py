# http/sse_api.py
import json, threading, time, urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any
import asyncio
from streams.broker import BROKER, Priority
from policy.enforcement import POLICY
from provenance.store import write_ledger

ADDR=("0.0.0.0", 8088)

# זיהוי משתמש בסיסי דרך header X-IMU-User
def _user(headers)->str: return headers.get("X-IMU-User","anon")

def _json(self:BaseHTTPRequestHandler, code:int, body:Dict[str,Any]):
    b=json.dumps(body).encode(); self.send_response(code)
    self.send_header("Content-Type","application/json")
    self.send_header("Content-Length", str(len(b))); self.end_headers()
    self.wfile.write(b)

def _bad(self:BaseHTTPRequestHandler, msg:str): _json(self, 400, {"error": msg})

def _ok(self:BaseHTTPRequestHandler, data:Dict[str,Any]): _json(self, 200, data)

class Handler(BaseHTTPRequestHandler):
    # /run_adapter?name=unity_build or k8s_deploy
    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        if u.path=="/run_adapter":
            user=_user(self.headers)
            qs=urllib.parse.parse_qs(u.query)
            name=qs.get("name",[""])[0]
            length=int(self.headers.get("Content-Length","0"))
            payload=json.loads(self.rfile.read(length) or b"{}")
            # אכיפת מדיניות: הרצת חיצוני?
            POLICY.require(user, need_external=True, need_evidence=True)
            # שלח אירוע התחלה
            asyncio.run(BROKER.publish("timeline", {"t":"start","adapter":name,"user":user,"ts":time.time()}, Priority.TELEMETRY))
            # הרצה בסדין
            import threading
            threading.Thread(target=self._run_sync, args=(user,name,payload), daemon=True).start()
            return _ok(self, {"status":"started","adapter":name})
        # /upload?name=foo.bin&sha256=...
        if u.path=="/upload":
            qs=urllib.parse.parse_qs(u.query)
            name=qs.get("name",[""])[0]; sha=qs.get("sha256",[""])[0]
            length=int(self.headers.get("Content-Length","0"))
            data=self.rfile.read(length)
            import os, hashlib
            calc=hashlib.sha256(data).hexdigest()
            if sha and sha!=calc: return _bad(self, "sha_mismatch")
            os.makedirs("./artifacts", exist_ok=True)
            path=f"./artifacts/{calc[:8]}_{name}"
            with open(path,"wb") as f: f.write(data)
            write_ledger({"type":"artifact","name":name,"sha256":calc,"ts":time.time(),"by":_user(self.headers),"path":path})
            asyncio.run(BROKER.publish("timeline", {"t":"artifact","name":name,"sha256":calc,"path":path,"ts":time.time()}, Priority.EVENTS))
            return _ok(self, {"ok":True,"sha256":calc,"path":path})
        return _bad(self, "unknown_endpoint")

    def _run_sync(self, user:str, name:str, payload:Dict[str,Any]):
        import adapters.unity_cli as unity_cli
        import adapters.k8s_deployer as k8s
        import adapters.cuda_runner as cuda
        try:
            if name=="unity_build":
                result = unity_cli.run_unity_build(payload)
            elif name=="k8s_deploy":
                result = k8s.deploy(payload)
            elif name=="cuda_job":
                result = cuda.run_job(payload)
            else:
                raise ValueError("unknown_adapter")
            asyncio.run(BROKER.publish("timeline", {"t":"done","adapter":name,"result":result,"ts":time.time()}, Priority.TELEMETRY))
        except Exception as e:
            asyncio.run(BROKER.publish("timeline", {"t":"error","adapter":name,"error":str(e),"ts":time.time()}, Priority.TELEMETRY))

    # SSE: /events?topic=timeline
    def do_GET(self):
        u=urllib.parse.urlparse(self.path)
        if u.path=="/events":
            qs=urllib.parse.parse_qs(u.query)
            topic=qs.get("topic",["timeline"])[0]
            user=_user(self.headers)
            p = POLICY.get(user)
            # אכיפת קצב לפי מדיניות
            self.send_response(200)
            self.send_header("Content-Type","text/event-stream")
            self.send_header("Cache-Control","no-cache")
            self.send_header("Connection","keep-alive")
            self.end_headers()
            loop = asyncio.new_event_loop()
            sid, q = BROKER.subscribe(topic)
            try:
                while True:
                    item = loop.run_until_complete(q.get())
                    line = f"data: {json.dumps(item)}\n\n".encode()
                    try:
                        self.wfile.write(line); self.wfile.flush()
                    except BrokenPipeError:
                        break
            finally:
                BROKER.unsubscribe(topic, sid)
            return
        if u.path=="/healthz":
            return _ok(self, {"ok":True})
        return _bad(self, "unknown_endpoint")

def serve_async():
    srv=ThreadingHTTPServer(ADDR, Handler)
    t=threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    return srv
