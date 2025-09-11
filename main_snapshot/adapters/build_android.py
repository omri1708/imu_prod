# adapters/android/build_android.py
import os, subprocess, tempfile, shutil
from typing import Dict, Any
from contracts.base import AdapterResult, require, ResourceRequired
from provenance import cas

def build(apk_out: str, project_dir: str, variant: str="debug") -> AdapterResult:
    # Preconditions
    try:
        require("gradle")
    except ResourceRequired:
        # Allow Android Gradle Wrapper instead
        if not os.path.exists(os.path.join(project_dir, "gradlew")):
            raise
    # Ensure JDK
    try: require("javac")
    except ResourceRequired as e:
        e.how_to = "Install JDK (e.g., Temurin). Ensure `javac` on PATH."
        raise e
    # Ensure Android SDK tools if using sdkmanager paths (best-effort)
    # Build
    env = os.environ.copy()
    logs = []
    cmd = ["./gradlew" if os.path.exists(os.path.join(project_dir,"gradlew")) else "gradle",
           f"assemble{variant.capitalize()}"]
    try:
        proc = subprocess.run(cmd, cwd=project_dir, capture_output=True, text=True, env=env, check=True)
        logs.append(proc.stdout)
    except subprocess.CalledProcessError as ex:
        return AdapterResult(False, logs=ex.stdout + "\n" + ex.stderr)
    # Locate APK
    apk = None
    for root, _, files in os.walk(project_dir):
        for f in files:
            if f.endswith(".apk"):
                apk = os.path.join(root, f)
    if not apk:
        return AdapterResult(False, logs="apk_not_found")
    shutil.copyfile(apk, apk_out)
    cid = cas.put_file(apk_out, {"type":"android_apk","variant":variant})
    return AdapterResult(True, artifact_path=apk_out, metrics={"size_bytes": os.path.getsize(apk_out)}, logs="\n".join(logs), provenance_cid=cid)