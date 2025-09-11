# adapters/unity.py
# -*- coding: utf-8 -*-
import os, subprocess, shutil
from typing import Tuple, List
from policy.policy_engine import PolicyViolation

def dry_run(project_path:str)->Tuple[bool,List[str]]:
    msgs=[]
    if not shutil.which("unity"):
        msgs.append("Unity CLI not found (install Unity Hub + CLI)")
    if not os.path.exists(os.path.join(project_path,"ProjectSettings")):
        msgs.append("Unity project structure missing")
    return (len(msgs)==0, msgs)

def build_standalone(project_path:str, out_dir:str)->str:
    ok, why = dry_run(project_path)
    if not ok: raise PolicyViolation("unity dry-run failed: " + "; ".join(why))
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "unity","-quit","-batchmode",
        "-projectPath", project_path,
        "-buildWindows64Player", os.path.join(out_dir,"Build.exe")
    ]
    subprocess.run(cmd, check=True)
    return os.path.join(out_dir,"Build.exe")