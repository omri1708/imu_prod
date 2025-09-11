# adapters/cuda/job_runner.py
import os, time
from typing import Callable, Dict, Any, Optional
from contracts.base import AdapterResult, ResourceRequired
from provenance import cas

def _has_nvidia_smi() -> bool:
    import shutil
    return shutil.which("nvidia-smi") is not None

def run_vector_add(n: int=1_000_000, use_gpu: bool=True) -> AdapterResult:
    import math, random
    # Try GPU via numba.cuda only if explicitly available; else CPU fallback.
    logs = []
    t0 = time.time()
    artifact = f"cuda_result_{int(t0)}.txt"
    ok = True
    try:
        if use_gpu:
            try:
                import numba
                from numba import cuda
                if not _has_nvidia_smi():
                    raise ResourceRequired("nvidia_driver","Install NVIDIA driver + CUDA runtime.")
                @cuda.jit
                def vadd(a,b,c):
                    i = cuda.grid(1)
                    if i < a.size: c[i] = a[i] + b[i]
                import numpy as np
                a = np.ones(n, dtype=np.float32)
                b = np.ones(n, dtype=np.float32)
                c = np.zeros(n, dtype=np.float32)
                d_a = cuda.to_device(a); d_b = cuda.to_device(b); d_c = cuda.to_device(c)
                threads = 256; blocks = (n + threads - 1)//threads
                vadd[blocks, threads](d_a, d_b, d_c)
                d_c.copy_to_host(c)
                checksum = float(c.sum())
                open(artifact,"w").write(f"gpu_checksum={checksum}\n")
                logs.append("gpu_ok")
            except ModuleNotFoundError:
                raise ResourceRequired("numba","pip install numba (or set use_gpu=False for CPU).")
        else:
            s = 0.0
            for _ in range(n): s += 2.0
            open(artifact,"w").write(f"cpu_checksum={s}\n")
            logs.append("cpu_ok")
    except ResourceRequired as e:
        ok = False
        logs.append(str(e))
    dt = time.time()-t0
    cid = cas.put_file(artifact, {"type":"cuda_job","n":n,"dt":dt}) if os.path.exists(artifact) else None
    return AdapterResult(ok, artifact_path=(artifact if os.path.exists(artifact) else None),
                         metrics={"dt_sec": dt}, logs="\n".join(logs), provenance_cid=cid)