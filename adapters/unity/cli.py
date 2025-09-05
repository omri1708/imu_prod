# adapters/unity/cli.py
# -*- coding: utf-8 -*-
import subprocess, shlex, os
from contracts.adapters import unity_env

def unity_batch(project_path: str, method: str, extra_args=None) -> None:
    unity_env()
    unity = "unity" if os.name != "nt" else "Unity.exe"
    args = extra_args or []
    cmd = [unity, "-quit", "-batchmode", "-nographics", "-projectPath", project_path, "-executeMethod", method] + args
    subprocess.check_call(" ".join(map(shlex.quote, cmd)), shell=True)