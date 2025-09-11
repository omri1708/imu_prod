# demos/cuda_job_end2end.py
from __future__ import annotations
import os, json, time, asyncio, shutil, urllib.request, websockets

HTTP=os.environ.get("IMU_API","http://127.0.0.1:8000")
WS=os.environ.get("IMU_WS","ws://127.0.0.1:8765")
USER=os.environ.get("IMU_USER","demo-user")

def _post(path, payload):
    req=urllib.request.Request(HTTP+path, method="POST", data=json.dumps(payload).encode(),
                               headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=20) as r: return json.loads(r.read().decode())

async def _push(note:str, pct:int=None):
    async with websockets.connect(WS) as ws:
        payload={"type":"event","ts":time.time(),"note":note}
        if pct is not None: payload={"type":"progress","ts":time.time(),"pct":pct,"note":note}
        await ws.send(json.dumps(payload))

def have(x)->bool: return shutil.which(x) is not None

async def run(src:str="kernel.cu", out:str="kernel.out"):
    await _push("cuda.e2e.start", 0)
    params={"src":src,"out":out}
    dry=_post("/adapters/dry_run", {"user_id":USER,"kind":"cuda.nvcc","params":params})
    await _push(f"cuda.dry.cmd={dry['cmd']}", 30)
    res=_post("/adapters/run", {"user_id":USER,"kind":"cuda.nvcc","params":params,"execute":have("nvcc")})
    await _push("cuda."+("compiled" if res["ok"] else "dry"), 100)
    await _push("cuda.e2e.finish")

if __name__=="__main__":
    import sys; asyncio.run(run(*(sys.argv[1:] if len(sys.argv)>1 else [])))