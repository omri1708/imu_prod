# imu_repo/tests/integration_micro.py
from __future__ import annotations
import os, sys, time, json, threading, http.server, socketserver, subprocess, tempfile
from typing import Dict, Any, List, Tuple
from urllib.request import urlopen, Request
from orchestration.services import Orchestrator, ServiceSpec, OrchestrationError
from obs.kpi import KPI

HELLO_PY = r"""
from http.server import BaseHTTPRequestHandler,HTTPServer
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200); self.end_headers(); self.wfile.write(b'OK'); return
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"hello":"imu"}')
HTTPServer(('127.0.0.1', 8011), H).serve_forever()
"""

SUM_PY = r"""
import json
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler,HTTPServer
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200); self.end_headers(); self.wfile.write(b'OK'); return
        q=parse_qs(urlparse(self.path).query)
        a=float(q.get('a',[0])[0]); b=float(q.get('b',[0])[0])
        s=str(a+b).encode()
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"sum":'+s+b'}')
HTTPServer(('127.0.0.1', 8012), H).serve_forever()
"""

def _write(tmpdir: str, name: str, content: str) -> str:
    path=os.path.join(tmpdir,name)
    with open(path,"w",encoding="utf-8") as f: f.write(content)
    return path

def run():
    tmp = os.path.abspath(".imu_state/micro")
    os.makedirs(tmp, exist_ok=True)
    hello = _write(tmp,"hello.py",HELLO_PY)
    summ  = _write(tmp,"sum.py",SUM_PY)

    orch = Orchestrator()
    kpi  = KPI()

    try:
        orch.start(ServiceSpec(name="hello", command=[sys.executable, hello],
                               http_health="http://127.0.0.1:8011/health",
                               tcp_health=("127.0.0.1",8011)))
        orch.start(ServiceSpec(name="sum", command=[sys.executable, summ],
                               http_health="http://127.0.0.1:8012/health",
                               tcp_health=("127.0.0.1",8012)))

        # קריאות אמתיות, רישום KPI
        t0=time.time()
        r = urlopen("http://127.0.0.1:8011/").read().decode()
        kpi.record(latency_ms=(time.time()-t0)*1000, error=False)

        t0=time.time()
        r2= urlopen("http://127.0.0.1:8012/?a=10&b=32").read().decode()
        kpi.record(latency_ms=(time.time()-t0)*1000, error=False)

        print("[integration] OK. hello:",r," sum:",r2)
        return 0
    except OrchestrationError as e:
        print("[integration] orchestration_failed:",e)
        kpi.record(latency_ms=0.0, error=True)
        return 1
    finally:
        orch.stop("sum"); orch.stop("hello")

if __name__=="__main__":
    raise SystemExit(run())
