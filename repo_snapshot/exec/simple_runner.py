# exec/simple_runner.py
from __future__ import annotations
import os, subprocess, shutil, time, sys, textwrap
from typing import Dict, Any

class ExecError(Exception): ...
class ResourceRequired(Exception):
    def __init__(self, what: str, how: str):
        super().__init__(f"resource_required:{what}")
        self.what=what; self.how=how

def which_any(names):
    for n in names:
        p = shutil.which(n)
        if p: return p
    return None

def run_python(code: str, workdir: str, filename: str="main.py", timeout_s: float=10.0) -> Dict[str,Any]:
    py = sys.executable or which_any(["python3","python"])
    if not py:
        raise ResourceRequired("python", "Install CPython 3.10+ and expose `python3`")
    os.makedirs(workdir, exist_ok=True)
    path = os.path.join(workdir, filename)
    with open(path,"w",encoding="utf-8") as f: f.write(textwrap.dedent(code))
    t0=time.time()
    try:
        p = subprocess.run([py, path], cwd=workdir, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        raise ExecError("timeout")
    return {"lang":"python","path":path,"exit":p.returncode,"stdout":p.stdout,"stderr":p.stderr,"elapsed_s":time.time()-t0}

def run_node(code: str, workdir: str, filename: str="main.mjs", timeout_s: float=10.0) -> Dict[str,Any]:
    node = which_any(["node"])
    if not node:
        raise ResourceRequired("node", "Install Node.js LTS and expose `node`")
    os.makedirs(workdir, exist_ok=True)
    path = os.path.join(workdir, filename)
    with open(path,"w",encoding="utf-8") as f: f.write(textwrap.dedent(code))
    t0=time.time()
    try:
        p = subprocess.run([node, path], cwd=workdir, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        raise ExecError("timeout")
    return {"lang":"node","path":path,"exit":p.returncode,"stdout":p.stdout,"stderr":p.stderr,"elapsed_s":time.time()-t0}
