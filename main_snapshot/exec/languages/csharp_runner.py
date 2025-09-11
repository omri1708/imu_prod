# imu_repo/exec/languages/csharp_runner.py
from __future__ import annotations
import os, subprocess, shutil, time
from typing import Dict, Any
from exec.errors import ResourceRequired, ExecError

CS_TPL = """
using System;
class Program {
  static void Main(string[] args) {
    // BEGIN
    %CODE%
    // END
  }
}
"""

def run(code: str, workdir: str, timeout_s: float = 20.0) -> Dict[str,Any]:
    dotnet = shutil.which("dotnet")
    if not dotnet:
        raise ResourceRequired("dotnet", "Install .NET SDK 7+ and expose `dotnet`")
    os.makedirs(workdir, exist_ok=True)
    proj = os.path.join(workdir, "app.csproj")
    with open(proj,"w",encoding="utf-8") as f:
        f.write("""<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net7.0</TargetFramework></PropertyGroup></Project>""")
    src = os.path.join(workdir, "Program.cs")
    with open(src,"w",encoding="utf-8") as f: f.write(CS_TPL.replace("%CODE%", code))
    t0=time.time()
    try:
        subprocess.check_call([dotnet,"build","-c","Release"], cwd=workdir, timeout=timeout_s)
        p = subprocess.run([dotnet,"run","-c","Release","--no-build"], cwd=workdir, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        raise ExecError("timeout")
    return {"lang":"csharp","exit":p.returncode,"stdout":p.stdout,"stderr":p.stderr,"elapsed_s":time.time()-t0,"artifact":src}
