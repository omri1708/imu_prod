# adapters/cuda/runner.py
# -*- coding: utf-8 -*-
import subprocess, shlex
from contracts.adapters import cuda_env

def run_cuda_job(py_entry: str, args=None):
    cuda_env()
    args = args or []
    cmd = ["python", py_entry] + list(args)
    subprocess.check_call(" ".join(map(shlex.quote, cmd)), shell=True)