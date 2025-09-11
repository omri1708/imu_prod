# demos/ios_build_end2end.py
from __future__ import annotations
import os, json, time, asyncio, shutil, platform, urllib.request, websockets

HTTP=os.environ.get("IMU_API","http://127.0.0.1:8000")
WS=os.environ.get("IMU_WS","ws://127.0.0.1:8765")
USER=os.environ.get("IMU_USER","demo-user")

def _post(path, payload):
    req=urllib.request.Request(HTTP+path, method="POST", data=json.dumps(payload).encode(),
                               headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=20) as r: return json.loads(r.read().decode())

async def _push(note:str):
    async with websockets.connect(WS) as ws:
        await ws.send(json.dumps({"type":"event","ts":time.time(),"note":note}))

def mac()->bool: return platform.system().lower()=="darwin"
def have(x)->bool: return shutil.which(x) is not None

async def run(xcworkspace:str, scheme:str="App", config:str="Release"):
    await _push("ios.e2e.start")
    params={"workspace":xcworkspace,"scheme":scheme,"config":config}
    dry=_post("/adapters/dry_run", {"user_id":USER,"kind":"ios.xcode","params":params})
    await _push(f"ios.dry.cmd={dry['cmd']}")
    do= mac() and have("xcodebuild")
    res=_post("/adapters/run", {"user_id":USER,"kind":"ios.xcode","params":params,"execute":bool(do)})
    await _push("ios.build."+("done" if res["ok"] else "dry"))
    await _push("ios.e2e.finish")

if __name__=="__main__":
    import sys
    if len(sys.argv)<2:
        print("usage: ios_build_end2end.py <.xcworkspace> [scheme] [configuration]")
        exit(2)
    asyncio.run(run(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else "App", sys.argv[3] if len(sys.argv)>3 else "Release"))