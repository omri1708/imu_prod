# adapters/android.py
# -*- coding: utf-8 -*-
import os, shutil, subprocess
from typing import Dict, Any, List
from common.exc import ResourceRequired
from adapters.base import BuildAdapter, BuildResult
from adapters.provenance_store import cas_put, evidence_for, register_evidence

from adapters.base import _need, run, put_artifact_text, evidence_from_text
from engine.adapter_types import AdapterResult

from storage.provenance import record_provenance


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