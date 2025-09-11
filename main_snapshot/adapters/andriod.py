# adapters/android.py
# -*- coding: utf-8 -*-
import os, shutil, subprocess, sys
from typing import Dict, List, Tuple
from policy.policy_engine import PolicyViolation

BREW = {
    "jdk": "openjdk@21",
    "android-sdk": "android-sdk",
    "gradle": "gradle"
}
WINGET = {
    "jdk": "EclipseAdoptium.Temurin.21.JDK",
    "android-sdk": "Google.AndroidSDK",
    "gradle": "Gradle.Gradle"
}
APT = {
    "jdk": "openjdk-21-jdk",
    "android-sdk": "android-sdk",
    "gradle": "gradle"
}

def _which(cmd:str)->bool: return shutil.which(cmd) is not None

def dry_run(project_dir:str)->Tuple[bool,List[str]]:
    cmds=[]
    if not _which("javac"): cmds.append("install JDK (Temurin 21)")
    if not _which("gradle"): cmds.append("install gradle")
    if not os.path.exists(os.path.join(project_dir,"app","build.gradle")):
        cmds.append("missing app/build.gradle")
    return (len(cmds)==0,cmds)

def build_apk(project_dir:str, out_dir:str)->str:
    ok, why = dry_run(project_dir)
    if not ok: raise PolicyViolation("android dry-run failed: " + "; ".join(why))
    os.makedirs(out_dir, exist_ok=True)
    # Gradle assemble (no placeholders)
    cmd = ["gradle","assembleDebug","-p",project_dir,"--no-daemon","-Dorg.gradle.jvmargs=-Xmx2g"]
    subprocess.run(cmd, check=True)
    # Discover APK
    apk = None
    for root,_,files in os.walk(project_dir):
        for f in files:
            if f.endswith(".apk"):
                apk=os.path.join(root,f)
    if not apk: raise RuntimeError("apk not found after build")
    dest = os.path.join(out_dir, os.path.basename(apk))
    shutil.copy2(apk,dest)
    return dest