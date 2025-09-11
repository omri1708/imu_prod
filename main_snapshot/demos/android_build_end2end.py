# demos/android_build_end2end.py
from __future__ import annotations
import os, json, time, asyncio, shutil
import websockets
import urllib.request

HTTP=os.environ.get("IMU_API","http://127.0.0.1:8000")
WS=os.environ.get("IMU_WS","ws://127.0.0.1:8765")
USER=os.environ.get("IMU_USER","demo-user")

def _post(path, payload):
    req=urllib.request.Request(HTTP+path, method="POST", data=json.dumps(payload).encode(),
                               headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=20) as r: return json.loads(r.read().decode())

async def _push(kind, data):
    async with websockets.connect(WS) as ws:
        data.setdefault("ts", time.time()); data.setdefault("type",kind)
        await ws.send(json.dumps(data))

def have(x:str)->bool: return shutil.which(x) is not None

async def run(app_dir:str):
    await _push("event", {"note":"android.e2e.start"})
    params={"flavor":"Release","buildType":"Aab","keystore":os.path.join(app_dir,"keystore.jks")}
    dry=_post("/adapters/dry_run", {"user_id":USER,"kind":"android.gradle","params":params})
    await _push("event", {"note":f"android.dry.cmd={dry['cmd']}"})
    exec_ok = have("gradle") or os.path.exists(os.path.join(app_dir,"gradlew"))
    res=_post("/adapters/run", {"user_id":USER,"kind":"android.gradle","params":params,"execute":bool(exec_ok)})
    await _push("progress", {"pct": 100 if res["ok"] else 60, "note": "android.build.done" if res["ok"] else "android.build.dry"})
    await _push("event", {"note":"android.e2e.finish"})

if __name__=="__main__":
    import sys, asyncio
    if len(sys.argv)<2:
        print("usage: android_build_end2end.py <app_dir>\n"
              "env IMU_API, IMU_WS")
        exit(2)
    asyncio.run(run(sys.argv[1]))