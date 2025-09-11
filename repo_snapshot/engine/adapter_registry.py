# engine/adapter_registry.py
import shutil, subprocess, os, sys, platform, json, time
from dataclasses import dataclass
from typing import Dict, Any
from evidence import cas
from common.errors import ResourceRequired

@dataclass
class AdapterResult:
    artifacts: Dict[str, str]  # logical_name -> file_path
    claims: list               # [{id, value, schema, evidence:[{sha256,...}]}]
    evidence: list             # [{sha256, uri?, fetched_at, ttl_sec, trust?}]

_ADAPTERS = {}

def register(kind: str):
    def deco(cls):
        _ADAPTERS[kind] = cls()
        return cls
    return deco

def get_adapter(kind: str):
    if kind not in _ADAPTERS:
        raise RuntimeError(f"unknown_adapter:{kind}")
    return _ADAPTERS[kind]

# ---------- K8s ----------
@register("k8s")
class K8sAdapter:
    def build(self, job: Dict[str,Any], user: str, ws: str, policy, ev_index) -> AdapterResult:
        kubectl = shutil.which("kubectl")
        kubeconfig = os.environ.get("KUBECONFIG") or os.path.expanduser("~/.kube/config")
        manifest = job.get("manifest") or "---\napiVersion: v1\nkind: Namespace\nmetadata:\n  name: imu\n"
        man_path = os.path.join(ws, "k8s-manifest.yaml")
        with open(man_path,"w",encoding="utf-8") as f: f.write(manifest)
        hash_manifest = cas.put_bytes(manifest.encode("utf-8"))

        if not kubectl or not os.path.exists(kubeconfig):
            raise ResourceRequired(
                "kubectl_or_kubeconfig_missing",
                need={"tool":"kubectl","how":"install kubectl + set KUBECONFIG",
                      "kubeconfig_path": kubeconfig, "manifest_cas": hash_manifest}
            )

        # apply בפועל
        cmd = [kubectl, "apply", "-f", man_path]
        cp = subprocess.run(cmd, capture_output=True, text=True)
        if cp.returncode != 0:
            raise RuntimeError(f"kubectl_apply_failed: {cp.stderr.strip()}")

        ev = {"sha256": hash_manifest, "uri": "cas://k8s/manifest", "fetched_at": time.time(), "ttl_sec": 30*24*3600, "trust": 0.8}
        claim = {"id":"deploy.k8s.apply", "value":{"ok":True}, "schema":{"type":"dict","properties":{"ok":{"type":"bool"}}}, "evidence":[{"sha256":hash_manifest}]}
        return AdapterResult(artifacts={"manifest": man_path}, claims=[claim], evidence=[ev])

# ---------- Android ----------
@register("android")
class AndroidAdapter:
    def build(self, job: Dict[str,Any], user: str, ws: str, policy, ev_index) -> AdapterResult:
        sdk = os.environ.get("ANDROID_SDK_ROOT") or os.environ.get("ANDROID_HOME")
        gradle = shutil.which("gradle") or shutil.which("./gradlew")
        if not sdk or not os.path.isdir(sdk) or not gradle:
            raise ResourceRequired(
                "android_sdk_or_gradle_missing",
                need={"ANDROID_SDK_ROOT": sdk or "$ANDROID_SDK_ROOT not set",
                      "tool":"gradle","how":"install Android SDK & Gradle (or include gradlew wrapper)"}
            )
        # דוגמה: הפקה של apk דרך gradle בעבודה קיימת (כאן ניצור פרויקט מזערי)
        proj = os.path.join(ws, "android_min")
        os.makedirs(proj, exist_ok=True)
        with open(os.path.join(proj,"build.gradle"),"w",encoding="utf-8") as f:
            f.write("// minimal placeholder build.gradle – supply your module\n")
        # בפועל יש להפעיל gradle assembleDebug בפרויקט אמיתי
        evsha = cas.put_json({"sdk":sdk, "gradle":gradle})
        claim = {"id":"build.android.gradle", "value":{"configured":True}, "schema":{"type":"dict","properties":{"configured":{"type":"bool"}}}, "evidence":[{"sha256":evsha}]}
        return AdapterResult(artifacts={"project_dir": proj}, claims=[claim], evidence=[{"sha256":evsha, "ttl_sec":365*24*3600, "fetched_at": time.time()}])

# ---------- iOS ----------
@register("ios")
class IOSAdapter:
    def build(self, job: Dict[str,Any], user: str, ws: str, policy, ev_index) -> AdapterResult:
        if platform.system() != "Darwin":
            raise ResourceRequired("xcode_only_on_macos", need={"os":"macOS","tool":"xcodebuild"})
        xcb = shutil.which("xcodebuild")
        if not xcb:
            raise ResourceRequired("xcodebuild_missing", need={"how":"Install Xcode Command Line Tools"})
        proj = os.path.join(ws, "ios_min")
        os.makedirs(proj, exist_ok=True)
        with open(os.path.join(proj,"Project.xcodeproj"),"w",encoding="utf-8") as f:
            f.write("// Xcode project placeholder – provide your sources\n")
        evsha = cas.put_json({"xcodebuild": xcb})
        claim = {"id":"build.ios.xcodebuild", "value":{"configured":True}, "schema":{"type":"dict","properties":{"configured":{"type":"bool"}}}, "evidence":[{"sha256":evsha}]}
        return AdapterResult(artifacts={"project_dir": proj}, claims=[claim], evidence=[{"sha256":evsha, "ttl_sec":365*24*3600, "fetched_at": time.time()}])

# ---------- Unity CLI ----------
@register("unity")
class UnityAdapter:
    def build(self, job: Dict[str,Any], user: str, ws: str, policy, ev_index) -> AdapterResult:
        unity = os.environ.get("UNITY_PATH") or shutil.which("Unity") or shutil.which("unity")
        if not unity:
            raise ResourceRequired("unity_cli_missing", need={"UNITY_PATH":"set to Unity executable", "how":"Install Unity + enable CLI"})
        proj = os.path.join(ws, "unity_min")
        os.makedirs(proj, exist_ok=True)
        with open(os.path.join(proj,"ProjectSettings.asset"),"w",encoding="utf-8") as f:
            f.write("%YAML 1.1\n# minimal settings\n")
        evsha = cas.put_json({"unity": unity})
        claim = {"id":"build.unity.cli", "value":{"configured":True}, "schema":{"type":"dict","properties":{"configured":{"type":"bool"}}}, "evidence":[{"sha256":evsha}]}
        return AdapterResult(artifacts={"project_dir": proj}, claims=[claim], evidence=[{"sha256":evsha, "ttl_sec":365*24*3600, "fetched_at": time.time()}])

# ---------- CUDA/GPU ----------
@register("cuda")
class CUDAAdapter:
    def build(self, job: Dict[str,Any], user: str, ws: str, policy, ev_index) -> AdapterResult:
        nvcc = shutil.which("nvcc")
        if not nvcc:
            raise ResourceRequired("nvcc_missing", need={"tool":"nvcc","how":"Install NVIDIA CUDA Toolkit"})
        src = job.get("source") or "__global__ void noop(){}"
        srcp = os.path.join(ws, "kernel.cu")
        with open(srcp,"w",encoding="utf-8") as f: f.write(src)
        outp = os.path.join(ws, "kernel.o")
        cp = subprocess.run([nvcc, "-c", srcp, "-o", outp], capture_output=True, text=True)
        if cp.returncode != 0:
            raise RuntimeError(f"nvcc_compile_failed: {cp.stderr.strip()}")
        evsha = cas.put_json({"nvcc": nvcc, "src_sha": cas.put_bytes(src.encode("utf-8"))})
        claim = {"id":"build.cuda.nvcc", "value":{"object_built":True}, "schema":{"type":"dict","properties":{"object_built":{"type":"bool"}}}, "evidence":[{"sha256":evsha}]}
        return AdapterResult(artifacts={"object": outp}, claims=[claim], evidence=[{"sha256":evsha, "ttl_sec":365*24*3600, "fetched_at": time.time()}])
