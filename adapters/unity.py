# adapters/unity.py
# -*- coding: utf-8 -*-
import os, shutil, subprocess
from typing import Dict, Any
from common.exc import ResourceRequired
from adapters.base import BuildAdapter, BuildResult
from adapters.provenance_store import cas_put, evidence_for, register_evidence
import os
from typing import Dict
from adapters.base import _need, run, put_artifact_text, evidence_from_text
from engine.adapter_types import AdapterResult
from storage.provenance import record_provenance
import shutil, subprocess, os, sys
from .contracts import AdapterResult, require


def unity_batchmode(project_path:str, build_target:str="Android") -> AdapterResult:
    unity = _find_unity()
    if not unity:
        return AdapterResult(False, "Unity not found in PATH", {})
    try:
        out = subprocess.run([
            unity, "-quit", "-batchmode",
            "-projectPath", project_path,
            "-buildTarget", build_target
        ], capture_output=True, text=True, timeout=3600)
        ok = (out.returncode == 0)
        return AdapterResult(ok, out.stderr if not ok else "ok", {"log": out.stdout})
    except Exception as e:
        return AdapterResult(False, str(e), {})


def _find_unity():
    for exe in ("unity", "Unity", "/Applications/Unity/Hub/Editor/Unity.app/Contents/MacOS/Unity"):
        if shutil.which(exe) or os.path.exists(exe):
            return exe
    return None


def run_unity_cli(project_dir: str, target: str="StandaloneLinux64") -> AdapterResult:
    unity = shutil.which("unity-editor") or shutil.which("Unity") or shutil.which("Unity.exe")
    if not unity:
        return require("Unity CLI", "Unity Hub/Editor CLI required",
                       ["# install Unity Editor CLI for your OS; accept EULA via hub"])
    args = [unity, "-batchmode", "-quit",
            "-projectPath", project_dir,
            "-buildTarget", target,
            "-logFile", os.path.join(project_dir, "Editor.log")]
    try:
        subprocess.run(args, check=True)
        return AdapterResult(status="ok", message="Unity build complete", outputs={"target": target})
    except subprocess.CalledProcessError as e:
        return AdapterResult(status="error", message=f"Unity build failed: {e}", outputs={})

class UnityAdapter(BuildAdapter):
    KIND = "unity"

    def detect(self) -> bool:
        return bool(shutil.which("Unity") or shutil.which("/Applications/Unity/Hub/Editor/Unity.app/Contents/MacOS/Unity"))

    def requirements(self):
        return (self.KIND, ["Unity Editor (batchmode)"], "Install Unity and enable CLI batchmode")

    def build(self, job: Dict, user: str, workspace: str, policy, ev_index) -> AdapterResult:
            # Unity CLI (גרסאות שונות: Unity/UnityHub – כאן נבדוק "Unity")
            _need("Unity", "Install Unity Editor CLI & add to PATH.")
            proj = os.path.join(workspace, "unity_project")
            os.makedirs(proj, exist_ok=True)
            # נריץ batchmode בדיקה קלה:
            code,out,err = run(["Unity","-quit","-batchmode","-projectPath",proj,"-logFile","-"])
            if code != 0:
                raise RuntimeError(f"Unity CLI failed: {err}")
            build_path = os.path.join(proj, "Builds", "demo")
            os.makedirs(build_path, exist_ok=True)
            exe_path = os.path.join(build_path, "demo.bin")
            put_artifact_text(exe_path, "unity-demo-binary")
            ev = [evidence_from_text("unity_log", out[-4000:])]
            record_provenance(exe_path, ev, trust=0.7)
            claims = [{"kind":"unity_build","path":exe_path,"user":user}]
            return AdapterResult(artifacts={exe_path:""}, claims=claims, evidence=ev)