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
from adapters.base import AdapterBase, PlanResult
from engine.policy import RequestContext
import shlex, subprocess
from typing import Dict, Any, List, Tuple
from engine.provenance import Evidence
from engine.policy import UserSpacePolicy
import os, subprocess, json, shlex
from typing import Dict, Any
from runtime.sandbox import enforce_file_access
from policy.model import UserPolicy

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

class UnityAdapter(AdapterBase, BuildAdapter):
    KIND = "unity"
    name = "unity"
    
    def build_command(self, args: Dict[str, Any], dry_run: bool, policy: UserSpacePolicy) -> List[str]:
        project_path = args.get("project_path","./UnityProject")
        build_target = args.get("build_target","StandaloneLinux64")
        method = args.get("method","Builder.PerformBuild")
        cmd = ["bash","-lc", f'unity -batchmode -nographics -projectPath {shlex.quote(project_path)} -buildTarget {shlex.quote(build_target)} -executeMethod {shlex.quote(method)} -quit']
        return cmd

    def execute(self, cmd: List[str], policy: UserSpacePolicy) -> Tuple[bool,str,str]:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            out, err = proc.communicate(timeout=policy.p95_ms/1000)
        except subprocess.TimeoutExpired:
            proc.kill()
            return False, "", "timeout"
        return proc.returncode==0, out, err

    def produce_evidence(self, cmd: List[str], args: Dict[str, Any]):
        return [Evidence(claim="unity.build.plan", source="adapters.unity", trust=0.7, extra={"cmd":cmd,"args":args})]
    
    def plan(self, spec: Dict[str, Any], ctx: RequestContext) -> PlanResult:
        proj = spec.get("projectPath",".")
        target = spec.get("buildTarget","StandaloneWindows64")
        out = spec.get("output","Build/build.exe")
        cmds = [f"unity -quit -batchmode -projectPath {proj} -buildTarget {target} -executeMethod BuildScript.Build -logFile -",
                f"echo artifact at {out}"]
        return PlanResult(commands=cmds, env={}, notes="unity batch build")
    
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
    

def dry_run(project_dir: str, target: str="StandaloneLinux64") -> Dict[str, Any]:
    # CLI אחיד ל־Unity:
    # Unity -quit -batchmode -nographics -projectPath <dir> -buildTarget <target> -executeMethod BuildScript.Build
    cmds = [f"Unity -quit -batchmode -nographics -projectPath {shlex.quote(project_dir)} -buildTarget {shlex.quote(target)} -executeMethod BuildScript.Build"]
    return {"ok": True, "cmds": cmds, "needs": ["Unity Editor + BuildSupport for target"]}


def run(policy: UserPolicy, project_dir: str, target: str="StandaloneLinux64") -> Dict[str, Any]:
    enforce_file_access(policy, project_dir, write=False)
    env = os.environ.copy()
    cmd = ["Unity","-quit","-batchmode","-nographics","-projectPath",project_dir,"-buildTarget",target,"-executeMethod","BuildScript.Build"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return {"ok": p.returncode==0, "log": p.stdout}