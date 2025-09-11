# demos/pack_a_ios_end2end.py
"""
iOS (xcodebuild) → dry/exec → evidence → WS live.
Requires macOS to actually execute; otherwise dry-run only.
"""
from __future__ import annotations
import os, json, time, asyncio, shutil, platform, urllib.request
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

def mac()->bool: return platform.system().lower()=="darwin"
def have(x)->bool: return shutil.which(x) is not None

async def run(xcworkspace:str, scheme:str="App", config:str="Release"):
    await _push("ios.pipeline.start",0)
    params={"workspace":xcworkspace,"scheme":scheme,"config":config}
    dry=_post("/adapters/dry_run", {"user_id":USER,"kind":"ios.xcode","params":params})
    await _push(f"ios.dry.cmd={dry['cmd']}",20)
    do= mac() and have("xcodebuild")
    res=_post("/adapters/run", {"user_id":USER,"kind":"ios.xcode","params":params,"execute":bool(do)})
    await _push("ios."+("exec" if res["ok"] else "dry"),100)

if __name__=="__main__":
    import sys
    if len(sys.argv)<2:
        print("usage: pack_a_ios_end2end.py <.xcworkspace> [scheme] [config]"); exit(2)
    asyncio.run(run(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else "App", sys.argv[3] if len(sys.argv)>3 else "Release"))