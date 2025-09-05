# adapters/android/build.py
# -*- coding: utf-8 -*-
import os, shutil, tempfile
from ..contracts import ensure_tool, run, record_provenance

import os, subprocess, shlex
from contracts.adapters import android_env
from engine.errors import ResourceRequired


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