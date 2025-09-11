# server/pipeline/run_adapter.py
from typing import Dict, Any
import shlex, asyncio
from server.security.provenance import ProvenanceStore
from server.events.bus import EventBus, Topic
from server.policy.enforcement import CapabilityPolicy
from server.state.ttl import TTLRules

class DryRunError(Exception): ...

async def run_adapter(payload: Dict[str, Any], dry: bool, event_bus: EventBus,
                      prov: ProvenanceStore, ttl: TTLRules, policy: CapabilityPolicy) -> Dict[str, Any]:
    """
    payload:
      {
        "adapter": "android"|"ios"|"unity"|"cuda"|"k8s",
        "action": "build"|"deploy"|"run",
        "args": {...},
        "require": ["android-sdk","unity-cli","k8s-cli", ...]    # optional
      }
    """
    adapter = payload.get("adapter")
    action = payload.get("action")
    args = payload.get("args", {})
    reqs = payload.get("require", [])

    # policy: request-needed capabilities (non-blocking)
    from server.capabilities.registry import capability_registry
    for cname in reqs:
        cap = capability_registry.resolve(cname)
        if not cap:
            raise DryRunError(f"unknown capability: {cname}")
        # If missing, emit telemetry + provenance; installation is done elsewhere via /capabilities/request
        if not cap.is_available():
            event_bus.emit(Topic.TELEMETRY, {"type":"capability_missing","capability":cname})
            prov.record_capability(cname, False, {"reason":"missing_at_run"})
    # dispatch to adapter
    if adapter == "android":
        plan = _plan_android(action, args)
    elif adapter == "ios":
        plan = _plan_ios(action, args)
    elif adapter == "unity":
        plan = _plan_unity(action, args)
    elif adapter == "cuda":
        plan = _plan_cuda(action, args)
    elif adapter == "k8s":
        plan = _plan_k8s(action, args)
    else:
        raise DryRunError(f"unknown adapter: {adapter}")

    # Provenance for transparency
    prov.record_adapter_plan(adapter, plan, dry=dry)
    event_bus.emit(Topic.TELEMETRY, {"type":"plan", "adapter": adapter, "plan": plan})

    if dry:
        return plan

    # Fake execution with clear transparency (this is where actual subprocess would run)
    event_bus.emit(Topic.TELEMETRY, {"type":"exec_start","adapter":adapter,"action":action})
    await asyncio.sleep(0.1)
    result = {"adapter": adapter, "action": action, "ok": True, "stdout": f"Simulated {adapter}.{action}"}
    event_bus.emit(Topic.TELEMETRY, {"type":"exec_done","adapter":adapter,"action":action,"ok":True})

    prov.record_adapter_run(adapter, result)
    return result

def _plan_android(action: str, a: Dict[str, Any]) -> Dict[str, Any]:
    proj = a.get("project_dir","/workspace/android")
    cmd = ["gradle", "assembleRelease"] if action == "build" else ["gradle","test"]
    return {"cmd": cmd, "cwd": proj, "env": {"ANDROID_HOME":"${ANDROID_HOME}"}}

def _plan_ios(action: str, a: Dict[str, Any]) -> Dict[str, Any]:
    ws = a.get("workspace","App.xcworkspace")
    scheme = a.get("scheme","App")
    sdk = a.get("sdk","iphoneos")
    cmd = ["xcodebuild","-workspace",ws,"-scheme",scheme,"-sdk",sdk,"build"]
    return {"cmd": cmd, "cwd": a.get("project_dir","."), "env": {}}

def _plan_unity(action: str, a: Dict[str, Any]) -> Dict[str, Any]:
    proj = a.get("project_dir","/workspace/unity")
    build_target = a.get("target","Android")
    out = a.get("output","/workspace/builds/game.apk")
    cmd = ["unity","-quit","-batchmode","-projectPath",proj,"-buildTarget",build_target,"-executeMethod","BuildScript.Build","-customBuildPath",out]
    return {"cmd": cmd, "cwd": proj, "env": {}}

def _plan_cuda(action: str, a: Dict[str, Any]) -> Dict[str, Any]:
    cu = a.get("file","kernel.cu")
    out = a.get("out","kernel.out")
    cmd = ["nvcc", cu, "-o", out]
    return {"cmd": cmd, "cwd": a.get("cwd","."), "env": {}}

def _plan_k8s(action: str, a: Dict[str, Any]) -> Dict[str, Any]:
    if action == "deploy":
        manifest = a.get("manifest","k8s/deploy.yaml")
        cmd = ["kubectl","apply","-f", manifest]
    elif action == "delete":
        manifest = a.get("manifest","k8s/deploy.yaml")
        cmd = ["kubectl","delete","-f", manifest]
    else:
        cmd = ["kubectl","get","pods","-o","wide"]
    return {"cmd": cmd, "cwd": a.get("cwd","."), "env": {}}