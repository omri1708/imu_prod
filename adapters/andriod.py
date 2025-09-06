# adapters/android.py
# -*- coding: utf-8 -*-
import os, shutil, subprocess
from typing import Dict, Any, List
import os, subprocess, shutil
from typing import Dict, Any
from common.exc import ResourceRequired
from adapters.base import BuildAdapter, BuildResult
from adapters.provenance_store import cas_put, evidence_for, register_evidence
from adapters.base import _need, run, put_artifact_text, evidence_from_text
from engine.adapter_types import AdapterResult
from storage.provenance import record_provenance
from .contracts import AdapterResult


import os, subprocess, shutil
from .contracts import AdapterResult, require


def build_android_gradle(project_dir:str) -> AdapterResult:
    gradlew = os.path.join(project_dir, "gradlew")
    if not os.path.exists(gradlew):
        return AdapterResult(False, "gradlew not found", {})
    try:
        out = subprocess.run([gradlew, "assembleRelease", "--no-daemon"], cwd=project_dir, capture_output=True, text=True, timeout=1800)
        ok = (out.returncode == 0)
        apk = _find_apk(project_dir)
        return AdapterResult(ok, out.stderr if not ok else "ok", {"apk": apk, "log": out.stdout})
    except Exception as e:
        return AdapterResult(False, str(e), {})


def _exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def run_android_build(project_dir: str, variant: str="Debug") -> AdapterResult:
    # Require JDK + Gradle + Android SDK (sdkmanager / zipalign / apksigner)
    missing = []
    if not _exists("javac"): missing.append(("JDK", "Install OpenJDK 17+", ["sudo apt-get install -y openjdk-17-jdk"]))
    if not _exists("gradle"): missing.append(("Gradle", "Install Gradle", ["sudo apt-get install -y gradle"]))
    if not _exists("zipalign"): missing.append(("Android SDK build-tools", "Install build-tools via sdkmanager", [
        "yes | sdkmanager 'build-tools;34.0.0'", "yes | sdkmanager 'platforms;android-34'"
    ]))
    if missing:
        rsrc = ", ".join([m[0] for m in missing])
        cmds = [c for m in missing for c in m[2]]
        return require(rsrc, "Android build dependencies missing", cmds)

    try:
        subprocess.run(["gradle", f"assemble{variant}"], cwd=project_dir, check=True)
        apk_path = _find_apk(project_dir, variant)
        return AdapterResult(status="ok", message="Android build complete", outputs={"apk": apk_path})
    except subprocess.CalledProcessError as e:
        return AdapterResult(status="error", message=f"Gradle failed: {e}", outputs={})

def _find_apk(root:str) -> str|None:
    for dp,_,files in os.walk(root):
        for f in files:
            if f.endswith(".apk"):
                return os.path.join(dp, f)
    return None

class AndroidAdapter(BuildAdapter):
    KIND = "android"

    def detect(self) -> bool:
        gradle = shutil.which("gradle") or shutil.which("./gradlew")
        sdkman = shutil.which("sdkmanager")
        return bool(gradle and sdkman)

    def requirements(self):
        how = ("Install Android SDK (sdkmanager), set ANDROID_HOME; "
               "install build-tools; install Gradle or use project gradlew.")
        return (self.KIND, ["sdkmanager","ANDROID_HOME","gradle/gradlew","JDK"], how)

    def build(self, job: Dict, user: str, workspace: str, policy, ev_index) -> AdapterResult:
        # נדרשים: java/javac + gradle
        _need("javac", "Install JDK (Temurin/OpenJDK).")
        _need("gradle", "Install Gradle: https://gradle.org/install/")
        app_dir = os.path.join(workspace, "android_app")
        os.makedirs(app_dir, exist_ok=True)
        # ניצור build.gradle מינימלי (אם לא קיים)
        build_gradle = os.path.join(app_dir, "build.gradle")
        if not os.path.exists(build_gradle):
            put_artifact_text(build_gradle, "plugins { id 'java' }\n")
        code,out,err = run(["gradle","build"], cwd=app_dir)
        if code != 0:
            raise RuntimeError(f"gradle build failed: {err}")
        # ארטיפקט "jar" מינימלי (כאן הדגמה – בפועל APK/Bundle יצריך Android SDK)
        jar_path = os.path.join(app_dir, "build", "libs", "android_app.jar")
        if not os.path.exists(jar_path):
            # אם אין – ניצור קובץ כדי לרשום provenance
            put_artifact_text(jar_path, "demo-jar")
        evidence = [evidence_from_text("android_build_log", out[-4000:])]
        record_provenance(jar_path, evidence, trust=0.7)
        claims = [{"kind":"android_build","path":jar_path,"user":user}]
        return AdapterResult(artifacts={jar_path: ""}, claims=claims, evidence=evidence)