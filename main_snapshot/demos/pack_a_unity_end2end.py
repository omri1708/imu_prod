# demos/pack_a_unity_end2end.py
"""
Unity → dry/exec build → evidence → optional docker build/push (skipped) → k8s deploy (dry/exec)
→ sends live progress/events to WS broker at ws://...:8765
Requires: server/http_api.py running on http://127.0.0.1:8000 and server/ws_progress.py on ws://127.0.0.1:8765
"""
from __future__ import annotations
import os, json, time, asyncio, shutil, urllib.request
import websockets  # pip install websockets

API = os.environ.get("IMU_API","http://127.0.0.1:8000")
WS  = os.environ.get("IMU_WS","ws://127.0.0.1:8765")
USER= os.environ.get("IMU_USER","demo-user")

def _post(path:str, obj:dict)->dict:
    req=urllib.request.Request(API+path, method="POST", data=json.dumps(obj).encode(),
                               headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())

async def _push(kind:str, note:str, pct:float|None=None):
    async with websockets.connect(WS) as ws:
        ev={"type": "progress" if pct is not None else "event", "ts": time.time(), "note": note}
        if pct is not None: ev["pct"]=pct
        await ws.send(json.dumps(ev, ensure_ascii=False))

def _have(cmd:str)->bool: return shutil.which(cmd) is not None

async def run(project_dir:str, target:str="Android", namespace:str="default", name:str="unity-app"):
    await _push("event","unity.pipeline.start",0)
    # 1) unity build dry
    params={"project":project_dir,"target":target,"method":"Builder.PerformBuild","version":"2022.3.44f1","log":"/tmp/unity.log"}
    dry=_post("/adapters/dry_run", {"user_id":USER,"kind":"unity.build","params":params})
    await _push("event", f"unity.dry.cmd={dry['cmd']}", 5)
    exec_ok=_have("unity") or _have("Unity") or _have("unity-editor")
    res=_post("/adapters/run", {"user_id":USER,"kind":"unity.build","params":params,"execute":bool(exec_ok)})
    await _push("event", "unity."+("exec" if res["ok"] else "dry"), 55)

    # 2) k8s deploy dry/exec
    manifest=f"""
apiVersion: apps/v1
kind: Deployment
metadata: {{name: {name}, namespace: {namespace}}}
spec:
  replicas: 1
  selector: {{matchLabels: {{app: {name}}}}}
  template:
    metadata: {{labels: {{app: {name}}}}}
    spec:
      containers:
      - name: web
        image: nginx:alpine
"""
    kd=_post("/adapters/dry_run", {"user_id":USER,"kind":"k8s.kubectl.apply","params":{"manifest":manifest,"namespace":namespace}})
    await _push("event", f"k8s.dry.cmd={kd['cmd']}", 70)
    kexec=_have("kubectl")
    kr=_post("/adapters/run", {"user_id":USER,"kind":"k8s.kubectl.apply",
                               "params":{"manifest":manifest,"namespace":namespace},"execute":bool(kexec)})
    await _push("event", "k8s."+("deployed" if kr["ok"] else "dry"), 100)
    await _push("event","unity.pipeline.finish")

if __name__=="__main__":
    import sys
    if len(sys.argv)<2:
        print("usage: pack_a_unity_end2end.py <UnityProjectDir> [target]")
        raise SystemExit(2)
    asyncio.run(run(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else "Android"))