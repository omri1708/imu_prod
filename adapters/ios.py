# adapters/ios.py
# -*- coding: utf-8 -*-
import os, shutil, subprocess
from typing import Dict, Any
from common.exc import ResourceRequired
from adapters.base import BuildAdapter, BuildResult
from adapters.provenance_store import cas_put, evidence_for, register_evidence

from adapters.base import _need, run, put_artifact_text, evidence_from_text
from engine.adapter_types import AdapterResult
from storage.provenance import record_provenance
from .contracts import AdapterResult, require


def run_ios_build(project_dir: str, scheme: str, sdk: str="iphoneos") -> AdapterResult:
    if not shutil.which("xcodebuild"):
        return require("Xcode", "Xcode Command Line Tools / Xcode.app required",
                       ["xcode-select --install"])
    try:
        subprocess.run(["xcodebuild", "-scheme", scheme, "-sdk", sdk, "build"],
                       cwd=project_dir, check=True)
        return AdapterResult(status="ok", message="iOS build complete", outputs={"xcbuild": "ok"})
    except subprocess.CalledProcessError as e:
        return AdapterResult(status="error", message=f"xcodebuild failed: {e}", outputs={})
    

def build_ios_xcodeproj(project_path:str, scheme:str, sdk:str="iphoneos") -> AdapterResult:
    if not os.path.exists(project_path):
        return AdapterResult(False, "xcodeproj not found", {})
    try:
        out = subprocess.run([
            "xcodebuild", "-project", project_path, "-scheme", scheme, "-sdk", sdk, "build"
        ], capture_output=True, text=True, timeout=1800)
        ok = (out.returncode == 0)
        return AdapterResult(ok, out.stderr if not ok else "ok", {"log": out.stdout})
    except Exception as e:
        return AdapterResult(False, str(e), {})

class IOSAdapter(BuildAdapter):
    KIND = "ios"

    def detect(self) -> bool:
        return bool(shutil.which("xcodebuild"))

    def requirements(self):
        return (self.KIND, ["xcodebuild","Xcode/CLT"], "Install Xcode command line tools and accept licenses")

    def build(self, job: Dict, user: str, workspace: str, policy, ev_index) -> AdapterResult:
        _need("xcodebuild", "Install Xcode / Command Line Tools (macOS only).")
        proj_dir = os.path.join(workspace, "ios_app")
        os.makedirs(proj_dir, exist_ok=True)
        # הדבקה מינימלית; פרויקט Xcode אמיתי דורש יצירה מלאה של scheme/targets
        # כאן נריץ xcodebuild -showsdks לבדיקה
        code,out,err = run(["xcodebuild","-showsdks"], cwd=proj_dir)
        if code != 0:
            raise RuntimeError(f"xcodebuild failed: {err}")
        app_path = os.path.join(proj_dir, "build", "ios_app.app")
        put_artifact_text(app_path, "demo-ios-app")
        ev = [evidence_from_text("ios_sdks", out[-4000:])]
        record_provenance(app_path, ev, trust=0.7)
        claims = [{"kind":"ios_build","path":app_path,"user":user}]
        return AdapterResult(artifacts={app_path:""}, claims=claims, evidence=ev)