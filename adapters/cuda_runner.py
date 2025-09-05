# adapters/cuda_runner.py
from __future__ import annotations
from .contracts.base import ResourceRequired, ProcessFailed, require_binary, run, sha256_file, BuildResult, ensure_dir, CAS_STORE
from adapters.contracts.base import record_event
from engine.progress import EMITTER
from perf.measure import measure, JOB_PERF
import os, tempfile, textwrap
from engine.errors import ResourceRequired
import os, shutil, subprocess, json, time

def run_job(payload:dict)->dict:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise ResourceRequired("cuda_toolkit",
            "NVIDIA CUDA Toolkit required (nvcc). Install from NVIDIA site.", True)
    src = payload.get("code", r"""
#include <stdio.h>
__global__ void add(int *a, int *b, int *c){ c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x]; }
int main(){ int a[1]={1}, b[1]={2}, c[1]={0}; int *d_a,*d_b,*d_c; 
cudaMalloc((void**)&d_a, sizeof(int)); cudaMalloc((void**)&d_b,sizeof(int)); cudaMalloc((void**)&d_c,sizeof(int));
cudaMemcpy(d_a,a,sizeof(int),cudaMemcpyHostToDevice); cudaMemcpy(d_b,b,sizeof(int),cudaMemcpyHostToDevice);
add<<<1,1>>>(d_a,d_b,d_c); cudaMemcpy(c,d_c,sizeof(int),cudaMemcpyDeviceToHost);
printf("%d\n", c[0]); return 0; }
""")
    with open("./cuda_job.cu","w") as f: f.write(src)
    t0=time.time()
    c = subprocess.run([nvcc, "./cuda_job.cu", "-o","./cuda_job"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if c.returncode!=0: raise RuntimeError(c.stdout[-4000:])
    r = subprocess.run(["./cuda_job"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dt=int((time.time()-t0)*1000)
    return {"ok":True,"ms":dt,"stdout":r.stdout.strip()}


def run_cuda_kernel(code: str, kernel: str, grid: tuple[int,int,int]=(1,1,1), block: tuple[int,int,int]=(1,1,1)) -> str:
    EMITTER.emit("timeline", {"phase":"cuda.prepare","kernel":kernel,"grid":grid,"block":block})
    require_binary("nvcc","Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads","CUDA compiler required")
    tmp=tempfile.mkdtemp(prefix="cuda-"); cu=os.path.join(tmp,"kernel.cu"); binp=os.path.join(tmp,"a.out")
    with open(cu,"w") as f: f.write(textwrap.dedent(code))
    (_, compile_dt) = measure(run, ["nvcc", cu, "-o", binp], None, None, 600)
    env=os.environ.copy(); env["CUDA_GRID"]=f"{grid[0]},{grid[1]},{grid[2]}"; env["CUDA_BLOCK"]=f"{block[0]},{block[1]},{block[2]}"
    (out, run_dt) = measure(run, [binp], None, env, 600)
    JOB_PERF.add(run_dt)
    EMITTER.emit("metrics", {"kind":"cuda.run","compile_secs":compile_dt,"run_secs":run_dt, **JOB_PERF.snapshot()})
    EMITTER.emit("timeline", {"phase":"cuda.done"})
    return out