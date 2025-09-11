# adapters/registry.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import subprocess, os, time, json, pathlib
from typing import Dict, Any, Optional, List
import platform, shutil, subprocess, os

from ..policy.user_policy import UserPolicy
from ..policy.policy_enforcer import PolicyEnforcer, PolicyViolation

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

# מיפוי קונקרטי (דוגמאות נפוצות, ניתן להרחיב):
WINGET_IDS = {
    "android-sdk": "Google.AndroidStudio",   # מתקין IDE + sdkmanager (לשינוי אם רוצים רק sdk)
    "unity-hub": "UnityTechnologies.UnityHub",
    "nodejs": "OpenJS.NodeJS",
    "go": "GoLang.Go",
    "cuda": "Nvidia.CUDA",
    "kubernetes-cli": "Kubernetes.kubectl"
}
BREW_FORMULAE = {
    "android-sdk": "android-commandlinetools",
    "unity-hub":  "unity-hub",
    "nodejs":     "node",
    "go":         "go",
    "cuda":       "cuda",
    "kubernetes-cli": "kubectl"
}

def _os_family():
    sys = platform.system().lower()
    if "windows" in sys:
        return "windows"
    if "darwin" in sys:
        return "mac"
    return "linux"

def _tool_exists(bin_name: str) -> bool:
    return shutil.which(bin_name) is not None

def _cmd_exists(cmd: List[str]) -> bool:
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except Exception:
        return False

def dry_run_install(capability: str) -> Dict[str, Any]:
    osfam = _os_family()
    if osfam == "windows":
        pkg = WINGET_IDS.get(capability)
        if not pkg:
            return {"ok": False, "reason":"unknown_capability"}
        return {"ok": True, "cmd":["winget","install","-e","--id",pkg]}
    elif osfam == "mac":
        pkg = BREW_FORMULAE.get(capability)
        if not pkg:
            return {"ok": False, "reason":"unknown_capability"}
        return {"ok": True, "cmd":["brew","install",pkg]}
    else:
        # Linux – דוגמאות נפוצות
        apt = shutil.which("apt")
        if apt and capability=="nodejs":
            return {"ok": True, "cmd":["bash","-lc","curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash - && sudo apt-get install -y nodejs"]}
        if apt and capability=="go":
            return {"ok": True, "cmd":["bash","-lc","sudo apt-get update && sudo apt-get install -y golang"]}
        if capability=="kubernetes-cli":
            return {"ok": True, "cmd":["bash","-lc","curl -LO https://storage.googleapis.com/kubernetes-release/release/`curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt`/bin/linux/amd64/kubectl && chmod +x kubectl && sudo mv kubectl /usr/local/bin/"]}

        return {"ok": False, "reason":"unsupported_linux_distro"}

def request_capability_install(capability: str, platform_hint: Optional[str]=None) -> Dict[str, Any]:
    """
    מרכיב פקודת התקנה *בלי להריץ*, כדי שתוכל לאשר/להריץ תחת CI/Agent שלך.
    """
    dr = dry_run_install(capability)
    return dr

def check_policy_for_adapter(policy: UserPolicy, enforcer: PolicyEnforcer, adapter_name: str, required_hosts: Optional[List[str]]=None):
    if not policy.can_use_capability(adapter_name):
        raise PolicyViolation(f"Adapter '{adapter_name}' disabled by policy.")
    for h in (required_hosts or []):
        enforcer.enforce_network_host(policy, h)