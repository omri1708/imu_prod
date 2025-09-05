# adapters/registry.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import subprocess, os, time, json, pathlib
from typing import Dict, Any

class ResourceRequired(Exception):
    def __init__(self, requirements:Dict[str,str]):
        super().__init__("resource_required")
        self.requirements = requirements

class AdapterRegistry:
    def __init__(self):
        self._adapters = {
            "unity.build": self._unity_build,
            "k8s.deploy": self._k8s_deploy,
        }

    def run(self, name:str, *, run_id:str, args:Dict[str,Any], user_id:str) -> Dict[str,Any]:
        if name not in self._adapters:
            raise ValueError(f"unknown_adapter:{name}")
        return self._adapters[name](run_id=run_id, args=args, user_id=user_id)

    # --- Unity Build (דורש Unity CLI קיים) ---
    def _unity_build(self, *, run_id:str, args:Dict[str,Any], user_id:str) -> Dict[str,Any]:
        unity_path = os.getenv("UNITY_CLI") or args.get("unity_cli")
        project_path = args["project_path"]
        out_path = args.get("out_path","./Build/Standalone")
        if not unity_path or not os.path.exists(unity_path):
            raise ResourceRequired({"UNITY_CLI":"Install Unity Editor with CLI; set UNITY_CLI=/path/to/Unity"})
        cmd = [
            unity_path, "-quit", "-batchmode",
            "-projectPath", project_path,
            "-buildWindows64Player", os.path.join(out_path, "game.exe"),
            "-nographics", "-logFile", f"unity_{run_id}.log"
        ]
        start = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"unity_build_failed: {res.stderr[-800:]}")
        return {"built": True, "out_path": out_path, "elapsed_s": time.time()-start}

    # --- K8s Deploy (דורש kubectl + context) ---
    def _k8s_deploy(self, *, run_id:str, args:Dict[str,Any], user_id:str) -> Dict[str,Any]:
        manifest = args["manifest_yaml"]
        tmp = pathlib.Path(f"./k8s_{run_id}.yaml")
        tmp.write_text(manifest, encoding="utf-8")
        kubectl = os.getenv("KUBECTL","kubectl")
        # בדיקת קיום:
        try:
            chk = subprocess.run([kubectl,"version","--client"], capture_output=True)
            if chk.returncode != 0:
                raise FileNotFoundError
        except Exception:
            raise ResourceRequired({"kubectl":"Install kubectl and configure KUBECONFIG/context"})
        res = subprocess.run([kubectl,"apply","-f",str(tmp)], capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"kubectl_apply_failed: {res.stderr[-800:]}")
        return {"deployed": True, "kubectl_stdout": res.stdout[-800:]}

ADAPTERS = AdapterRegistry()