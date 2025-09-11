# imu_repo/exec/languages/java_runner.py
from __future__ import annotations
import os, subprocess, shutil, textwrap, time
from typing import Dict, Any
from exec.errors import ResourceRequired, ExecError

JAVA_TPL = """
public class Main {
    public static void main(String[] args) throws Exception {
        %CODE%
    }
}
"""

def run(code: str, workdir: str, timeout_s: float = 15.0) -> Dict[str,Any]:
    javac = shutil.which("javac"); java = shutil.which("java")
    if not (javac and java):
        raise ResourceRequired("java", "Install JDK 17+ (javac/java) and expose on PATH")
    os.makedirs(workdir, exist_ok=True)
    src = os.path.join(workdir, "Main.java")
    with open(src,"w",encoding="utf-8") as f:
        f.write(JAVA_TPL.replace("%CODE%", code))
    t0=time.time()
    try:
        subprocess.check_call([javac, src], cwd=workdir, timeout=timeout_s)
        p = subprocess.run([java, "Main"], cwd=workdir, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        raise ExecError("timeout")
    return {"lang":"java","exit":p.returncode,"stdout":p.stdout,"stderr":p.stderr,"elapsed_s":time.time()-t0,"artifact":src}
