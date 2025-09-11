# adapters/android_builder.py
from __future__ import annotations
from .contracts.base import ResourceRequired, ProcessFailed, require_binary, run, sha256_file, BuildResult, ensure_dir, CAS_STORE
from adapters.contracts.base import record_event
import os, re
from engine.progress import EMITTER
from perf.measure import measure, BUILD_PERF



def build_android_gradle(project_dir: str, variant: str="Release", gradlew: str="./gradlew") -> BuildResult:
    EMITTER.emit("timeline", {"phase":"android.prepare","project":project_dir,"variant":variant})
    require_binary("javac","Install JDK (sdkman/brew/apt)","JDK required for Android builds")
    if not os.path.isfile(os.path.join(project_dir,"gradlew")):
        require_binary("gradle","Install Gradle: https://gradle.org/install/","gradle wrapper not found")
        gradlew="gradle"
    if os.environ.get("ANDROID_HOME") is None and os.environ.get("ANDROID_SDK_ROOT") is None and not os.path.exists(os.path.join(project_dir,"local.properties")):
        raise ResourceRequired("Android SDK","Install Android SDK + set ANDROID_HOME/ANDROID_SDK_ROOT","Android SDK not configured")

    (out, dt) = measure(run, [gradlew, f"assemble{variant}"], project_dir, None, 7200)
    BUILD_PERF.add(dt)
    EMITTER.emit("metrics", {"kind":"android.build","project":project_dir,"variant":variant,"secs":dt, **BUILD_PERF.snapshot()})
    EMITTER.emit("progress", {"project":project_dir,"pct":90,"msg":"Scanning outputs"})

    outputs=[]
    for root,_,files in os.walk(os.path.join(project_dir,"app","build","outputs")):
        for f in files:
            if f.endswith((".apk",".aab")): outputs.append(os.path.join(root,f))
    if not outputs: 
        record_event("android.no_outputs", {"dir":project_dir,"log_tail":out[-2000:]})
        raise ProcessFailed([gradlew, f"assemble{variant}"], 0, out, "No APK/AAB produced")

    artifact=max(outputs, key=os.path.getmtime)
    digest=CAS_STORE.put_file(artifact)
    EMITTER.emit("timeline", {"phase":"android.artifact","path":artifact,"sha256":digest})
    record_event("artifact.store", {"platform":"android","path":artifact,"sha256":digest})
    return BuildResult(artifact=artifact, sha256=digest, meta={"variant":variant,"tool":"gradle"})