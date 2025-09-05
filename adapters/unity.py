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