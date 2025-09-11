# adapters/cuda/runner.py
# -*- coding: utf-8 -*-
import subprocess, shlex
import os
from typing import Dict, Any
from provenance.audit import AuditLog


def run_cuda_job(cfg: Dict[str,Any], audit: AuditLog):
    src = cfg["source"]
    out = cfg["output_bin"]
    nvcc = os.environ.get("NVCC_PATH","nvcc")
    # compile
    cmd_compile = f'{shlex.quote(nvcc)} -O3 {shlex.quote(src)} -o {shlex.quote(out)}'
    audit.append("adapter.cuda","compile",{"cmd":cmd_compile})
    subprocess.check_call(cmd_compile, shell=True)
    # run
    run_args = " ".join(shlex.quote(a) for a in (cfg.get("run_args") or []))
    cmd_run = f'{shlex.quote(out)} {run_args}'
    audit.append("adapter.cuda","run",{"cmd":cmd_run})
    subprocess.check_call(cmd_run, shell=True)
    audit.append("adapter.cuda","success",{"bin":out})
    return {"ok": True, "binary": out}
