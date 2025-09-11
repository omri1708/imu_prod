# synth/test.py
from __future__ import annotations
import subprocess, time, os, http.client, socket
from typing import Dict, Any, List, Tuple
from exec.simple_runner import run_python, run_node, ExecError, ResourceRequired

def _free_port(start=18080) -> int:
    p = start
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                p+=1

def start_service(language: str, entry: str, timeout_s: float = 3.0) -> Tuple[subprocess.Popen, int]:
    port = _free_port()
    env = os.environ.copy()
    env["PORT"] = str(port)
    if language=="python":
        p = subprocess.Popen([os.sys.executable, entry], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    else:
        node = "node"
        p = subprocess.Popen([node, entry], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # wait for "SERVING"
    t0=time.time()
    ok=False
    while time.time()-t0<timeout_s:
        line = p.stdout.readline().strip()
        if "SERVING" in line:
            ok=True; break
        time.sleep(0.05)
    if not ok:
        try: p.kill()
        except Exception: pass
        raise ExecError("service_start_failed")
    return p, port

def http_get(port: int, path: str="/") -> Tuple[int, str]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
    conn.request("GET", path)
    r = conn.getresponse()
    body = r.read().decode("utf-8","replace")
    return r.status, body

def run_tests(language: str, entry: str, checks: List[Dict[str,Any]]) -> Dict[str,Any]:
    p, port = start_service(language, entry)
    results=[]
    try:
        for c in checks:
            st, body = http_get(port, c["path"])
            ok = (st==c["expect_status"]) and (c.get("expect_contains","") in body)
            results.append({"path":c["path"],"status":st,"ok":ok,"body":body[:120]})
    finally:
        try: p.terminate()
        except Exception: pass
    passed = all(r["ok"] for r in results)
    return {"port":port, "results":results, "passed":passed}