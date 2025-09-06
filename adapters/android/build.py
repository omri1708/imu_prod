# adapters/android/build.py
# -*- coding: utf-8 -*-
import os, subprocess, shlex
from typing import Dict, Any

from ..contracts import ensure_tool, run, record_provenance
from contracts.adapters import android_env
from engine.errors import ResourceRequired
from provenance.audit import AuditLog


def run_android_build(cfg: Dict[str,Any], audit: AuditLog):
    proj = cfg["project_dir"]
    variant = cfg["variant"]
    task = cfg.get("gradle_task","assemble")
    gradlew = os.path.join(proj, "gradlew")
    if not os.path.exists(gradlew):
        raise RuntimeError("gradlew_not_found")
    cmd = f'{shlex.quote(gradlew)} {task}{variant.capitalize()}'
    env = dict(os.environ)
    if cfg.get("keystore"):
        env["IMU_KEYSTORE"] = cfg["keystore"]
        env["IMU_KEYALIAS"] = cfg.get("keystore_alias","")
        env["IMU_KEYPASS"] = cfg.get("keystore_pass","")
    audit.append("adapter.android","invoke",{"cmd":cmd,"proj":proj})
    subprocess.check_call(cmd, cwd=proj, shell=True, env=env)
    audit.append("adapter.android","success",{"variant":variant})
    return {"ok": True, "artifact_hint": os.path.join(proj, "app","build","outputs")}

def build_apk(project_dir: str, variant: str = "Release") -> str:
    android_env()
    gradlew = os.path.join(project_dir, "gradlew")
    if os.path.isfile(gradlew):
        cmd = f"{shlex.quote(gradlew)} assemble{variant}"
    else:
        cmd = f"gradle assemble{variant}"
    subprocess.check_call(cmd, cwd=project_dir, shell=True)
    # מצא APK
    out = os.path.join(project_dir, "app","build","outputs","apk", variant.lower())
    for root,_,files in os.walk(out):
        for f in files:
            if f.endswith(".apk"): return os.path.join(root,f)
    raise FileNotFoundError("APK not found after build")


def build_android(project_dir: str, variant: str = "Debug") -> dict:
    """
    בונה APK ע"י gradle wrapper אם קיים, אחרת gradle.
    דרישות: JDK + Android SDK/Build Tools מותקנים בסביבת המשתמש.
    """
    # כלים נדרשים
    ensure_tool("javac", "Install JDK (e.g. temurin) and ensure javac on PATH")
    # gradle/gradlew
    gradlew = os.path.join(project_dir, "gradlew")
    if os.path.exists(gradlew):
        cmd = [gradlew, f"assemble{variant}"]
    else:
        ensure_tool("gradle", "Install Gradle and ensure gradle on PATH")
        cmd = ["gradle", f"assemble{variant}"]
    out = run(cmd, cwd=project_dir)
    # מציאת APK
    apk = None
    for root, _, files in os.walk(project_dir):
        for f in files:
            if f.endswith(".apk"):
                apk = os.path.join(root, f)
    if not apk:
        raise RuntimeError("APK not found after build")
    prov = record_provenance("android_build", {"project": project_dir, "variant": variant}, apk)
    return {"apk": apk, "provenance": prov.__dict__, "log": out}