# adapters/unity/cli.py
# -*- coding: utf-8 -*-
import subprocess, shlex, os
from contracts.adapters import unity_env
from typing import Dict, Any
from provenance.audit import AuditLog


def run_unity_cli(cfg: Dict[str,Any], audit: AuditLog):
    project = cfg["project_path"]
    target = cfg["build_target"]
    out = cfg["output_path"]
    unity = os.environ.get("UNITY_PATH","/Applications/Unity/Hub/Editor/Current/Unity.app/Contents/MacOS/Unity")
    if not os.path.exists(unity):
        unity = "unity"  # hope in PATH
    args = [
        shlex.quote(unity),
        "-batchmode","-quit",
        "-nographics",
        "-projectPath", shlex.quote(project),
        "-buildTarget", shlex.quote(target),
        "-logFile","-",
    ]
    if target == "StandaloneOSX":
        args += ["-buildOSXUniversalPlayer", shlex.quote(out)]
    elif target == "StandaloneWindows64":
        args += ["-buildWindows64Player", shlex.quote(out)]
    elif target == "WebGL":
        args += ["-executeMethod","BuildScript.BuildWebGL", "-buildPath", shlex.quote(out)]
    elif target == "Android":
        args += ["-executeMethod","BuildScript.BuildAndroid", "-buildPath", shlex.quote(out)]
    elif target == "iOS":
        args += ["-executeMethod","BuildScript.BuildiOS", "-buildPath", shlex.quote(out)]
    for s in (cfg.get("custom_args") or []):
        args.append(shlex.quote(s))
    cmd = " ".join(args)
    audit.append("adapter.unity","invoke",{"cmd":cmd})
    subprocess.check_call(cmd, shell=True, cwd=project)
    audit.append("adapter.unity","success",{"output": out})
    return {"ok": True, "artifact_hint": out}


def unity_batch(project_path: str, method: str, extra_args=None) -> None:
    unity_env()
    unity = "unity" if os.name != "nt" else "Unity.exe"
    args = extra_args or []
    cmd = [unity, "-quit", "-batchmode", "-nographics", "-projectPath", project_path, "-executeMethod", method] + args
    subprocess.check_call(" ".join(map(shlex.quote, cmd)), shell=True)