# adapters/cuda.py
# -*- coding: utf-8 -*-
import os, subprocess, shutil
from typing import Tuple, List
from policy.policy_engine import PolicyViolation

def dry_run()->Tuple[bool,List[str]]:
    msgs=[]
    if not shutil.which("nvcc"): msgs.append("nvcc not found (CUDA toolkit)")
    return (len(msgs)==0, msgs)

def compile_kernel(cu_path:str, out_so:str)->str:
    ok, why = dry_run()
    if not ok: raise PolicyViolation("cuda dry-run failed: " + "; ".join(why))
    cmd = ["nvcc","-shared","-Xcompiler","-fPIC", cu_path, "-o", out_so]
    os.makedirs(os.path.dirname(out_so), exist_ok=True)
    subprocess.run(cmd, check=True)
    return out_so