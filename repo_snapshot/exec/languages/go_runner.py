# imu_repo/exec/languages/go_runner.py
from __future__ import annotations
import os, subprocess, shutil, textwrap, time
from typing import Dict, Any
from exec.errors import ResourceRequired, ExecError

def run(code: str, workdir: str, timeout_s: float = 12.0) -> Dict[str,Any]:
    go = shutil.which("go")
    if not go:
        raise ResourceRequired("go", "Install Go 1.20+ and expose `go`")
    os.makedirs(workdir, exist_ok=True)
    main = os.path.join(workdir, "main.go")
    with open(main,"w",encoding="utf-8") as f: f.write(textwrap.dedent(code))
    t0=time.time()
    try:
        p = subprocess.run([go, "run", main], cwd=workdir, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        raise ExecError("timeout")
    return {"lang":"go","exit":p.returncode,"stdout":p.stdout,"stderr":p.stderr,"elapsed_s":time.time()-t0,"artifact":main}
