# demos/pack_a_k8s_end2end.py
"""
K8s → dry/exec apply (deployment) → WS live.
"""
from __future__ import annotations
import os, json, time, asyncio, urllib.request, shutil
import websockets

API=os.environ.get("IMU_API","http://127.0.0.1:8000")
WS=os.environ.get("IMU_WS","ws://127.0.0.1:8765")
USER=os.environ.get("IMU_USER","demo-user")

def _post(p,o):
    req=urllib.request.Request(API+p, method="POST", data=json.dumps(o).encode(), headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=20) as r: return json.loads(r.read().decode())

async def _push(note,pct=None):
    async with websockets.connect(WS) as ws:
        e={"type":"progress" if pct is not None else "event","ts":time.time(),"note":note}
        if pct is not None: e["pct"]=pct
        await ws.send(json.dumps(e))

def have(x)->bool: return shutil.which(x) is not None

async def run(image="nginx:alpine", ns="default"):
    await _push("k8s.pipeline.start",0)
    man=f"""
apiVersion: apps/v1
kind: Deployment
metadata: {{name: imu-demo, namespace: {ns}}}
spec:
  replicas: 1
  selector: {{matchLabels: {{app: imu-demo}}}}
  template:
    metadata: {{labels: {{app: imu-demo}}}}
    spec:
      containers:
      - name: web
        image: {image}
        ports: [{{containerPort: 80}}]
"""
    d=_post("/adapters/dry_run", {"user_id":USER,"kind":"k8s.kubectl.apply","params":{"manifest":man,"namespace":ns}})
    await _push(f"k8s.dry.cmd={d['cmd']}", 50)
    r=_post("/adapters/run", {"user_id":USER,"kind":"k8s.kubectl.apply","params":{"manifest":man,"namespace":ns},"execute":have("kubectl")})
    await _push("k8s."+("exec" if r["ok"] else "dry"),100)

if __name__=="__main__":
    import sys; asyncio.run(run(*(sys.argv[1:] if len(sys.argv)>1 else [])))