# adapters/cuda.py
# -*- coding: utf-8 -*-
import os, shutil, subprocess, tempfile
from typing import Dict, Any
from common.exc import ResourceRequired
from adapters.base import BuildAdapter, BuildResult
from adapters.provenance_store import cas_put, evidence_for, register_evidence

import subprocess, shlex
from typing import Dict, Any, List, Tuple
from engine.provenance import Evidence
from engine.policy import UserSpacePolicy
from adapters.base import _need, run, put_artifact_text, evidence_from_text
from engine.adapter_types import AdapterResult
from storage.provenance import record_provenance
from .contracts import AdapterResult, require
from adapters.base import AdapterBase, PlanResult

from engine.policy import RequestContext

def run_cuda_job(script:str) -> AdapterResult:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return AdapterResult(False, "nvcc not found", {})
    try:
        out = subprocess.run(["bash","-lc", script], capture_output=True, text=True, timeout=7200)
        ok = (out.returncode == 0)
        return AdapterResult(ok, out.stderr if not ok else "ok", {"log": out.stdout})
    except Exception as e:
        return AdapterResult(False, str(e), {})

class CUDAAdapter(AdapterBase, BuildAdapter):
    KIND = "cuda"
    name = "cuda"

    def build_command(self, args: Dict[str, Any], dry_run: bool, policy: UserSpacePolicy) -> List[str]:
        script = args.get("script","/usr/local/bin/run_cuda_job.sh")
        job_args = args.get("job_args",[])
        sh = " ".join(shlex.quote(str(x)) for x in job_args)
        cmd = ["bash","-lc", f"{shlex.quote(script)} {sh}"]
        return cmd

    def execute(self, cmd: List[str], policy: UserSpacePolicy) -> Tuple[bool,str,str]:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            out, err = proc.communicate(timeout=policy.p95_ms/1000)
        except subprocess.TimeoutExpired:
            proc.kill()
            return False, "", "timeout"
        return proc.returncode==0, out, err

    def produce_evidence(self, cmd: List[str], args: Dict[str, Any]):
        return [Evidence(claim="cuda.job.plan", source="adapters.cuda", trust=0.7, extra={"cmd":cmd,"args":args})]


    def plan(self, spec: Dict[str, Any], ctx: RequestContext) -> PlanResult:
        src = spec.get("src","kernel.cu")
        out = spec.get("out","kernel.out")
        arch = spec.get("arch","sm_86")
        cmds = [f"nvcc -arch={arch} {src} -o {out}"]
        return PlanResult(commands=cmds, env={}, notes="nvcc compile")
    
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