# adapters/unity/build.py
# -*- coding: utf-8 -*-
import os, shutil, subprocess
from adapters.contracts import ResourceRequired

def _find_unity():
    # נסה unity-editor ב־PATH; אם לא, מקובל במק/לינוקס
    cand = ["unity-editor", "/Applications/Unity/Hub/Editor/2021.3.0f1/Unity.app/Contents/MacOS/Unity"]
    for p in cand:
        if shutil.which(p) or os.path.exists(p):
            return p
    raise ResourceRequired("Unity Editor CLI", "Install Unity Editor and ensure CLI path (unity-editor)")

def unity_build(project_dir: str, build_target: str = "Android", log_file: str = "unity_build.log"):
    u = _find_unity()
    cmd = [u, "-batchmode", "-quit", "-projectPath", project_dir,
           "-buildTarget", build_target, "-logFile", log_file]
    subprocess.run(cmd, check=True)
    return {"ok": True, "log": os.path.abspath(log_file)}