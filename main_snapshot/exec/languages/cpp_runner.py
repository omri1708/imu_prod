# imu_repo/exec/languages/cpp_runner.py
from __future__ import annotations
import os, subprocess, shutil, time, textwrap
from typing import Dict, Any
from exec.errors import ResourceRequired, ExecError

CPP_TPL = r"""
#include <bits/stdc++.h>
using namespace std;
int main(){ 
    %CODE%
    return 0; 
}
"""

def run(code: str, workdir: str, timeout_s: float = 15.0) -> Dict[str,Any]:
    gpp = shutil.which("g++")
    if not gpp:
        raise ResourceRequired("g++", "Install GCC/G++ and expose `g++`")
    os.makedirs(workdir, exist_ok=True)
    src = os.path.join(workdir, "main.cpp")
    binp = os.path.join(workdir, "a.out")
    with open(src,"w",encoding="utf-8") as f: f.write(CPP_TPL.replace("%CODE%", textwrap.dedent(code)))
    t0=time.time()
    try:
        subprocess.check_call([gpp, src, "-O2", "-std=c++17", "-o", binp], cwd=workdir, timeout=timeout_s)
        p = subprocess.run([binp], cwd=workdir, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        raise ExecError("timeout")
    return {"lang":"cpp","exit":p.returncode,"stdout":p.stdout,"stderr":p.stderr,"elapsed_s":time.time()-t0,"artifact":src}
