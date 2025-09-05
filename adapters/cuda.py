# adapters/cuda.py
# -*- coding: utf-8 -*-
import os, shutil, subprocess, tempfile
from typing import Dict, Any
from common.exc import ResourceRequired
from adapters.base import BuildAdapter, BuildResult
from adapters.provenance_store import cas_put, evidence_for, register_evidence
import os
from typing import Dict
from adapters.base import _need, run, put_artifact_text, evidence_from_text
from engine.adapter_types import AdapterResult
from storage.provenance import record_provenance
import shutil, subprocess, os
from .contracts import AdapterResult, require


def run_cuda_job(cuda_src: str, out_bin: str) -> AdapterResult:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return require("CUDA Toolkit (nvcc)", "Install NVIDIA CUDA toolkit matching your driver",
                       ["# visit: https://developer.nvidia.com/cuda-downloads",
                        "# then ensure nvcc on PATH"])
    try:
        subprocess.run([nvcc, cuda_src, "-o", out_bin], check=True)
        return AdapterResult(status="ok", message="CUDA compile ok", outputs={"bin": out_bin})
    except subprocess.CalledProcessError as e:
        return AdapterResult(status="error", message=f"nvcc failed: {e}", outputs={})

class CUDAAdapter(BuildAdapter):
    KIND = "cuda"

    def detect(self) -> bool:
        return bool(shutil.which("nvcc"))

    def requirements(self):
        return (self.KIND, ["CUDA Toolkit (nvcc)"], "Install NVIDIA CUDA toolkit and drivers")

    def build(self, job: Dict, user: str, workspace: str, policy, ev_index) -> AdapterResult:
        _need("nvidia-smi", "Install NVIDIA drivers.")
        _need("nvcc", "Install CUDA Toolkit.")
        kernels = os.path.join(workspace, "cuda_kernels")
        os.makedirs(kernels, exist_ok=True)
        cu = os.path.join(kernels, "axpy.cu")
        if not os.path.exists(cu):
            put_artifact_text(cu, r"""
extern "C" __global__ void axpy(float a, const float* x, float* y, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) y[i] = a * x[i] + y[i];
}
""")
        out_path = os.path.join(kernels, "axpy.ptx")
        code,out,err = run(["nvcc","-ptx",cu,"-o",out_path], cwd=kernels)
        if code != 0:
            raise RuntimeError(f"nvcc failed: {err}")
        ev = [evidence_from_text("cuda_nvcc_out", out[-4000:])]
        record_provenance(out_path, ev, trust=0.85)
        claims = [{"kind":"cuda_kernel","path":out_path,"user":user}]
        return AdapterResult(artifacts={out_path:""}, claims=claims, evidence=ev)