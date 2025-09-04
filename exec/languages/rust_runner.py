# imu_repo/exec/languages/rust_runner.py
from __future__ import annotations
import os, subprocess, shutil, time, textwrap
from typing import Dict, Any
from exec.errors import ResourceRequired, ExecError

RS_TPL = r"""
fn main(){
    %CODE%
}
"""

def run(code: str, workdir: str, timeout_s: float = 20.0) -> Dict[str,Any]:
    rustc = shutil.which("rustc")
    if not rustc:
        raise ResourceRequired("rustc", "Install Rust toolchain and expose `rustc`")
    os.makedirs(workdir, exist_ok=True)
    src = os.path.join(workdir, "main.rs")
    binp = os.path.join(workdir, "main")
    with open(src,"w",encoding="utf-8") as f: f.write(RS_TPL.replace("%CODE%", textwrap.dedent(code)))
    t0=time.time()
    try:
        subprocess.check_call([rustc, src, "-O", "-o", binp], cwd=workdir, timeout=timeout_s)
        p = subprocess.run([binp], cwd=workdir, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        raise ExecError("timeout")
    return {"lang":"rust","exit":p.returncode,"stdout":p.stdout,"stderr":p.stderr,"elapsed_s":time.time()-t0,"artifact":src}
