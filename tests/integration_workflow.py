# imu_repo/tests/integration_workflow.py
from __future__ import annotations
import os, sys, json, time, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import urlopen, Request
from typing import Dict, Any, Optional
from orchestration.compose_workflow import ensure_redis_compose, shutdown_redis_compose
from adapters.redis_resp import RedisResp
from obs.kpi import KPI
from obs.alerts import AlertMonitor
from persistence.policy_store import PolicyStore

# Fallback queue (ללא Redis)
from adapters.db_localqueue import LocalQueue

QUEUE_KEY="imu:jobs"
RESULT_KEY="imu:results"

class JobQueue:
    def __init__(self):
        self._redis: Optional[RedisResp] = None
        self._local = LocalQueue(".imu_state/queue_workflow")
        self.use_redis=False
        try:
            if ensure_redis_compose(True):
                r=RedisResp()
                if r.ping():
                    self._redis=r; self.use_redis=True
        except Exception:
            self.use_redis=False

    def put(self, job: Dict[str,Any]):
        if self.use_redis:
            self._redis.lpush(QUEUE_KEY, json.dumps(job))
        else:
            self._local.put(json.dumps(job))

    def get(self, timeout_s:int=2) -> Optional[Dict[str,Any]]:
        if self.use_redis:
            v=self._redis.brpop(QUEUE_KEY, timeout_s)
            return json.loads(v) if v else None
        else:
            v=self._local.get(timeout_ms=timeout_s*1000)
            return json.loads(v) if v else None

    def set_result(self, job_id: str, result: Dict[str,Any]):
        if self.use_redis:
            self._redis.set(f"{RESULT_KEY}:{job_id}", json.dumps(result))
        else:
            self._local.put(json.dumps({"rid":job_id,"result":result}))

    def get_result(self, job_id: str) -> Optional[Dict[str,Any]]:
        if self.use_redis:
            v=self._redis.get(f"{RESULT_KEY}:{job_id}")
            return json.loads(v) if v else None
        else:
            # חיפוש נאיבי בתור המקומי (דמו אמיתי)
            for _ in range(10):
                data=self._local.get(timeout_ms=50)
                if data:
                    obj=json.loads(data)
                    if obj.get("rid")==job_id: return obj.get("result")
        return None

Q = JobQueue()
K = KPI()
P = PolicyStore()
A = AlertMonitor(K, P)

def worker_loop(stop_flag):
    while not stop_flag["stop"]:
        job=Q.get(timeout_s=1)
        if not job: continue
        # "עיבוד": סכימה פשוטה
        a=float(job.get("a",0)); b=float(job.get("b",0))
        res={"sum":a+b, "ts":time.time()}
        Q.set_result(job["id"], res)

class APIHandler(BaseHTTPRequestHandler):
    def _json(self, obj: Dict[str,Any], code:int=200):
        body=json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        t0=time.time()
        try:
            if self.path.startswith("/health"):
                self._json({"ok": True}); return
            if self.path.startswith("/result"):
                # /result?id=xyz
                from urllib.parse import urlparse, parse_qs
                q = parse_qs(urlparse(self.path).query)
                rid = q.get("id",[""])[0]
                r = Q.get_result(rid)
                if r is None:
                    self._json({"ready": False}, code=202)
                else:
                    self._json({"ready": True, "result": r})
                return
            self._json({"err":"not_found"}, 404)
        finally:
            K.record((time.time()-t0)*1000, False)

    def do_POST(self):
        t0=time.time(); err=False
        try:
            ln=int(self.headers.get("Content-Length","0"))
            obj=json.loads(self.rfile.read(ln).decode() or "{}")
            job_id=str(int(time.time()*1000))
            Q.put({"id":job_id, **obj})
            self._json({"accepted": True, "id": job_id}, 202)
        except Exception as e:
            err=True; self._json({"err": str(e)}, 500)
        finally:
            K.record((time.time()-t0)*1000, err)

def run_server(port:int=8020):
    httpd=HTTPServer(("127.0.0.1",port), APIHandler)
    print(f"[workflow] API on http://127.0.0.1:{port}")
    httpd.serve_forever()

def run():
    # התחל מוניטור התראות
    A.start()
    # התחל worker
    stop={"stop":False}
    wt=threading.Thread(target=worker_loop, args=(stop,), daemon=True); wt.start()
    # התחל API
    st=threading.Thread(target=run_server, kwargs={"port":8020}, daemon=True); st.start()
    time.sleep(1.0)

    # שלח בקשה
    import urllib.request
    req=urllib.request.Request("http://127.0.0.1:8020", method="POST",
                               data=json.dumps({"a":21,"b":34}).encode(),
                               headers={"Content-Type":"application/json"})
    r=urllib.request.urlopen(req).read().decode()
    rid=json.loads(r)["id"]

    # המתן לתוצאה
    for _ in range(50):
        resp=json.loads(urllib.request.urlopen(f"http://127.0.0.1:8020/result?id={rid}").read().decode())
        if resp.get("ready"): 
            print("[workflow] result:", resp["result"])
            break
        time.sleep(0.1)

    # ניקיון
    stop["stop"]=True
    time.sleep(0.5)
    A.stop()
    shutdown_redis_compose()
    return 0

if __name__=="__main__":
    raise SystemExit(run())
