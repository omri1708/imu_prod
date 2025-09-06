# demos/k8s_deploy_end2end.py
from __future__ import annotations
import os, json, time, asyncio, shutil, urllib.request, websockets, tempfile

HTTP=os.environ.get("IMU_API","http://127.0.0.1:8000")
WS=os.environ.get("IMU_WS","ws://127.0.0.1:8765")
USER=os.environ.get("IMU_USER","demo-user")

def _post(path, payload):
    req=urllib.request.Request(HTTP+path, method="POST", data=json.dumps(payload).encode(),
                               headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=20) as r: return json.loads(r.read().decode())

async def _push(kind:str, note:str, pct:int=None):
    async with websockets.connect(WS) as ws:
        obj={"type":kind,"ts":time.time(),"note":note}
        if pct is not None: obj["pct"]=pct
        await ws.send(json.dumps(obj))

def have(x)->bool: return shutil.which(x) is not None

async def run(image="nginx:alpine", ns="default"):
    await _push("event","k8s.e2e.start",5)
    man = f"""
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
    dry=_post("/adapters/dry_run", {"user_id":USER,"kind":"k8s.kubectl.apply","params":{"manifest":man,"namespace":ns}})
    await _push("event", f"k8s.dry.cmd={dry['cmd']}", 40)
    res=_post("/adapters/run", {"user_id":USER,"kind":"k8s.kubectl.apply","params":{"manifest":man,"namespace":ns},"execute":have("kubectl")})
    await _push("event","k8s."+("deployed" if res["ok"] else "dry"), 100)

if __name__=="__main__":
    import sys; asyncio.run(run(*(sys.argv[1:] if len(sys.argv)>1 else [])))