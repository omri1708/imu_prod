# imu_repo/exec/languages/python_runner.py
from __future__ import annotations
import os, subprocess, sys, tempfile, textwrap, time
from typing import Dict, Any
from exec.errors import ResourceRequired, ExecError

def run(code: str, workdir: str, timeout_s: float = 8.0) -> Dict[str,Any]:
    py = sys.executable
    if not py:
        raise ResourceRequired("python", "Install CPython 3.10+ and expose as `python3`")
    os.makedirs(workdir, exist_ok=True)
    path = os.path.join(workdir, "main.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(code))
    t0=time.time()
    try:
        p = subprocess.run([py, path], cwd=workdir, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        raise ExecError("timeout")
    dt=time.time()-t0
    return {"lang":"python","exit":p.returncode,"stdout":p.stdout,"stderr":p.stderr,"elapsed_s":dt,"artifact":path}
