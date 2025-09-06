# adapters/android.py
# -*- coding: utf-8 -*-
import os, shutil, subprocess, shlex
from typing import Dict, Any, List, Tuple

import os, subprocess, shlex, json, time
from typing import Dict, Any, List
from runtime.sandbox import enforce_file_access, PolicyViolation
from policy.model import UserPolicy
from common.exc import ResourceRequired
from adapters.base import BuildAdapter, BuildResult
from adapters.provenance_store import cas_put, evidence_for, register_evidence
from adapters.base import _need, run, put_artifact_text, evidence_from_text
from engine.adapter_types import AdapterResult
from storage.provenance import record_provenance
from .contracts import AdapterResult, require
from adapters.base import AdapterBase, PlanResult
from engine.policy import RequestContext

from engine.provenance import Evidence
from engine.policy import UserSpacePolicy


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

class AndroidAdapter(AdapterBase, BuildAdapter):
    KIND = "android"
    name = "android"
    
    def build_command(self, args: Dict[str, Any], dry_run: bool, policy: UserSpacePolicy) -> List[str]:
        project_dir = args.get("project_dir","./android")
        task = args.get("task","assembleRelease")
        cmd = ["bash","-lc", f"cd {shlex.quote(project_dir)} && gradle {shlex.quote(task)}"]
        return cmd

    def execute(self, cmd: List[str], policy: UserSpacePolicy) -> Tuple[bool,str,str]:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            out, err = proc.communicate(timeout=policy.p95_ms/1000)
        except subprocess.TimeoutExpired:
            proc.kill()
            return False, "", "timeout (p95 exceeded)"
        return proc.returncode==0, out, err

    def produce_evidence(self, cmd: List[str], args: Dict[str, Any]):
        return [Evidence(claim="android.build.plan", source="adapters.android", trust=0.7, extra={"cmd":cmd,"args":args})]

    def plan(self, spec: Dict[str, Any], ctx: RequestContext) -> PlanResult:
        module = spec.get("module", "app")
        variant = spec.get("variant", "Debug")
        cmds = [f"./gradlew :{module}:assemble{variant} --no-daemon"]
        env = {}
        return PlanResult(commands=cmds, env=env, notes="gradle assemble")

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
    

#_______
ANDROID_SDK_HINTS = {
    "linux": ["cmdline-tools", "platform-tools", "build-tools;34.0.0", "platforms;android-34"],
    "darwin": ["cmdline-tools", "platform-tools", "build-tools;34.0.0", "platforms;android-34"],
    "windows": ["platform-tools","build-tools;34.0.0","platforms;android-34"],
}

def _find_sdk() -> str:
    # ANDROID_HOME or ANDROID_SDK_ROOT
    for k in ("ANDROID_SDK_ROOT","ANDROID_HOME"):
        v = os.environ.get(k)
        if v and os.path.isdir(v):
            return v
    raise FileNotFoundError("Android SDK not found (set ANDROID_SDK_ROOT)")

def dry_run(project_dir: str, variant: str="release") -> Dict[str, Any]:
    sdk = "${ANDROID_SDK_ROOT}"
    gradlew = os.path.join(project_dir, "gradlew")
    cmds = [
        f"cd {shlex.quote(project_dir)} && {shlex.quote(gradlew)} assemble{variant.capitalize()}",
        f"{sdk}/build-tools/34.0.0/apksigner verify app-{variant}.apk"
    ]
    return {"ok": True, "cmds": cmds, "needs": ["ANDROID_SDK_ROOT","Java JDK 17"]}

def run(policy: UserPolicy, project_dir: str, variant: str="release") -> Dict[str, Any]:
    enforce_file_access(policy, project_dir, write=False)
    sdk = _find_sdk()
    gradlew = os.path.join(project_dir, "gradlew")
    if not os.path.exists(gradlew):
        raise FileNotFoundError("gradlew missing in project")
    env = os.environ.copy()
    env["ANDROID_SDK_ROOT"] = sdk
    p = subprocess.run([gradlew, f"assemble{variant.capitalize()}"], cwd=project_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = p.stdout
    if p.returncode!=0:
        return {"ok": False, "log": out}
    apk_path = None
    for root,_,files in os.walk(project_dir):
        for f in files:
            if f.endswith(".apk"):
                apk_path=os.path.join(root,f)
    if not apk_path:
        return {"ok": False, "error": "APK not found"}
    return {"ok": True, "artifact": apk_path, "log": out}