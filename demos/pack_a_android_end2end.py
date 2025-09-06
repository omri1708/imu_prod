# demos/pack_a_android_end2end.py
"""
Android (Gradle) → dry/exec → evidence → WS live progress.
"""
from __future__ import annotations
import os, json, time, asyncio, shutil, urllib.request
import websockets

API=os.environ.get("IMU_API","http://127.0.0.1:8000")
WS=os.environ.get("IMU_WS","ws://127.0.0.1:8765")
USER=os.environ.get("IMU_USER","demo-user")

def _post(path,obj):
    req=urllib.request.Request(API+path, method="POST", data=json.dumps(obj).encode(),
                               headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=20) as r: return json.loads(r.read().decode())

async def _push(note,pct=None):
    async with websockets.connect(WS) as ws:
        ev={"type":"progress" if pct is not None else "event","ts":time.time(),"note":note}
        if pct is not None: ev["pct"]=pct
        await ws.send(json.dumps(ev))

def have(x)->bool: return shutil.which(x) is not None

async def run(app_dir:str):
    await _push("android.pipeline.start",0)
    params={"flavor":"Release","buildType":"Aab","keystore":os.path.join(app_dir,"keystore.jks")}
    d=_post("/adapters/dry_run", {"user_id":USER,"kind":"android.gradle","params":params})
    await _push(f"android.dry.cmd={d['cmd']}",15)
    ok= have("gradle") or os.path.exists(os.path.join(app_dir,"gradlew"))
    r=_post("/adapters/run", {"user_id":USER,"kind":"android.gradle","params":params,"execute":bool(ok)})
    await _push("android."+("exec" if r["ok"] else "dry"), 100)

if __name__=="__main__":
    import sys
    if len(sys.argv)<2:
        print("usage: pack_a_android_end2end.py <app_dir>"); exit(2)
    asyncio.run(run(sys.argv[1]))