# adapters/mobile/android_build.py
# -*- coding: utf-8 -*-
import os, shutil, subprocess
from adapters.contracts import ResourceRequired

def find_gradle(project_dir: str):
    gw = os.path.join(project_dir, "gradlew")
    if os.path.exists(gw): return gw
    if shutil.which("gradle"): return "gradle"
    raise ResourceRequired("Gradle", "Install Gradle or include ./gradlew in the project")

def build_android(project_dir: str, task: str = "assembleRelease"):
    gw = find_gradle(project_dir)
    env = os.environ.copy()
    cmd = [gw, task]
    if gw.endswith("gradlew"): cmd = ["bash", gw, task]
    subprocess.run(cmd, cwd=project_dir, check=True, env=env)
    # פלט סטנדרטי ל־APK/ABB תחת app/build/outputs
    out_dir = os.path.join(project_dir, "app", "build", "outputs")
    return {"ok": True, "outputs_dir": out_dir}