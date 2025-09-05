# adapters/cuda_runner.py
from __future__ import annotations
from .contracts.base import ResourceRequired, ProcessFailed, require_binary, run, sha256_file, BuildResult, ensure_dir, CAS_STORE
from adapters.contracts.base import record_event
from engine.progress import EMITTER
from perf.measure import measure, JOB_PERF
import os, tempfile, textwrap

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