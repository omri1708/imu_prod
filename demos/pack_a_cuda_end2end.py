# demos/pack_a_cuda_end2end.py
"""
CUDA → dry/exec compile via nvcc → WS live.
"""
from __future__ import annotations
import os, json, time, asyncio, shutil, urllib.request
import websockets

API=os.environ.get("IMU_API","http://127.0.0.1:8000")
WS=os.environ.get("IMU_WS","ws://127.0.0.1:8765")
USER=os.environ.get("IMU_USER","demo-user")

def _post(p,o):
    r=urllib.request.Request(API+p, method="POST", data=json.dumps(o).encode(), headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(r, timeout=20) as h: return json.loads(h.read().decode())

async def _push(n,p=None):
    async with websockets.connect(WS) as ws:
        e={"type":"progress" if p is not None else "event","ts":time.time(),"note":n}
        if p is not None: e["pct"]=p
        await ws.send(json.dumps(e))

def have(x)->bool: return shutil.which(x) is not None

async def run(src="kern.cu", out="kern"):
    await _push("cuda.pipeline.start",0)
    d=_post("/adapters/dry_run", {"user_id":USER,"kind":"cuda.nvcc","params":{"src":src,"out":out}})
    await _push(f"cuda.dry.cmd={d['cmd']}",25)
    r=_post("/adapters/run", {"user_id":USER,"kind":"cuda.nvcc","params":{"src":src,"out":out},"execute":have("nvcc")})
    await _push("cuda."+("exec" if r["ok"] else "dry"),100)

if __name__=="__main__":
    import sys; asyncio.run(run(*(sys.argv[1:] if len(sys.argv)>1 else [])))