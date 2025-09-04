# imu_repo/exec/languages/node_runner.py
from __future__ import annotations
import os, subprocess, shutil, textwrap, time
from typing import Dict, Any
from exec.errors import ResourceRequired, ExecError

def run(code: str, workdir: str, timeout_s: float = 8.0) -> Dict[str,Any]:
    node = shutil.which("node")
    if not node:
        raise ResourceRequired("node", "Install Node.js LTS and expose `node`")
    os.makedirs(workdir, exist_ok=True)
    path = os.path.join(workdir, "main.mjs")
    with open(path,"w",encoding="utf-8") as f: f.write(textwrap.dedent(code))
    t0=time.time()
    try:
        p = subprocess.run([node, path], cwd=workdir, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        raise ExecError("timeout")
    return {"lang":"node","exit":p.returncode,"stdout":p.stdout,"stderr":p.stderr,"elapsed_s":time.time()-t0,"artifact":path}
