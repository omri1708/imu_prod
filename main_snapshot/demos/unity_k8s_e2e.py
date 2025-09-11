# demos/unity_k8s_end2end.py
"""
Unity → build (execute if CLI found else dry) → package (optional docker) → deploy to k8s (dry if kubectl missing)
→ push live progress to WS and to UI timeline.
"""
from __future__ import annotations
import os, json, time, asyncio, shutil, platform, tempfile, subprocess
from typing import Dict, Any
import websockets  # pip install websockets

HTTP = os.environ.get("IMU_API","http://127.0.0.1:8000")
WS   = os.environ.get("IMU_WS", "ws://127.0.0.1:8765")
USER = os.environ.get("IMU_USER","demo-user")

def _which(x:str)->bool:
    from shutil import which; return which(x) is not None

async def push(kind:str, data:Dict[str,Any]):
    async with websockets.connect(WS) as ws:
        msg={"type":kind,"ts":time.time(),"pct":data.get("pct"),"note":data.get("note")}
        await ws.send(json.dumps(msg, ensure_ascii=False))

def http_post(path:str, payload:dict) -> dict:
    import urllib.request
    req = urllib.request.Request(HTTP+path, method="POST",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))

async def run(project_dir:str, target:str="Android", k8s_ns:str="default", name:str="unity-web"):
    await push("event", {"note":"unity.e2e.start"})
    # 1) build unity (exec when possible)
    params={"project":project_dir,"target":target,"method":"Builder.PerformBuild","version":"2022.3.44f1","log":"/tmp/unity.log"}
    await push("progress", {"pct":5, "note":"unity.build.prepare"})
    dry = http_post("/adapters/dry_run", {"user_id":USER,"kind":"unity.build","params":params})
    await push("event", {"note": f"unity.dry.cmd={dry.get('cmd')}"})
    do_exec = _which("unity") or _which("Unity") or _which("unity-editor")
    res = http_post("/adapters/run", {"user_id":USER,"kind":"unity.build","params":params,"execute": bool(do_exec)})
    await push("progress", {"pct":50, "note": "unity.build.done" if res["ok"] else "unity.build.dry"})

    # 2) package to docker (optional)(skipped: environment dependent)
    await push("progress", {"pct":65, "note":"package.skip_or_external"})

    # 3) deploy to k8s (dry if kubectl missing)
    man = f"""
apiVersion: apps/v1
kind: Deployment
metadata: {{name: {name}, namespace: {k8s_ns}}}
spec:
  replicas: 1
  selector: {{matchLabels: {{app: {name}}}}}
  template:
    metadata: {{labels: {{app: {name}}}}}
    spec:
      containers:
      - name: web
        image: nginx:alpine
        ports: [{{containerPort: 80}}]
"""
    await push("progress", {"pct":75, "note":"k8s.apply.prepare"})
    kdry = http_post("/adapters/dry_run", {"user_id":USER,"kind":"k8s.kubectl.apply","params":{"manifest":man,"namespace":k8s_ns}})
    await push("event", {"note": f"k8s.dry.cmd={kdry.get('cmd')}"})
    do_k = _which("kubectl")
    kres = http_post("/adapters/run", {"user_id":USER,"kind":"k8s.kubectl.apply","params":{"manifest":man,"namespace":k8s_ns},"execute": bool(do_k)})
    await push("progress", {"pct":100, "note":"k8s.deploy.done" if kres["ok"] else "k8s.deploy.dry"})
    await push("event", {"note":"unity.e2e.finish"})

if __name__=="__main__":
    import sys
    if len(sys.argv)<2:
        print("usage: unity_k8s_end2end.py <UnityProjectDir> [target Android|WebGL|...]\n"
              "env: IMU_API=http://127.0.0.1:8000  IMU_WS=ws://127.0.0.1:8765")
        raise SystemExit(2)
    asyncio.run(run(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else "Android"))