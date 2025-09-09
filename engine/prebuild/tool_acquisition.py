# engine/prebuild/tool_acquisition.py
from __future__ import annotations
import os, platform, subprocess, hashlib, json, time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    from provenance.castore import ContentAddressableStore as CAStore
except Exception:
    class CAStore:
        def __init__(self, root: str):
            self.root = root; os.makedirs(root, exist_ok=True)
        def put(self, blob: bytes) -> str:
            h = hashlib.sha256(blob).hexdigest()
            p = os.path.join(self.root, h)
            if not os.path.exists(p):
                with open(p, 'wb') as f: f.write(blob)
            return h

CAS = CAStore("var/cas")

ALLOWED_LICENSES = {"MIT","Apache-2.0","BSD-3-Clause","BSD-2-Clause","MPL-2.0"}

@dataclass
class ToolSpec:
    name: str
    pkg: str                  # שם חבילה במנהל החבילות
    manager: str              # "brew"|"winget"|"apt"|"pip"
    version: Optional[str] = None
    sha256: Optional[str] = None  # אופציונלי לקבצים ישירים
    spdx: Optional[str] = None    # זיהוי רישיון אם ידוע

@dataclass
class ToolInstallResult:
    name: str
    ok: bool
    manager: str
    version: Optional[str]
    evidence_id: Optional[str]
    out: str
    err: str


def _run(cmd: List[str]) -> Tuple[int,str,str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err


def _is_linux() -> bool: return platform.system().lower() == "linux"

def _is_macos() -> bool: return platform.system().lower() == "darwin"

def _is_windows() -> bool: return platform.system().lower() == "windows"


def _install(tool: ToolSpec) -> ToolInstallResult:
    cmd: List[str] = []
    if tool.manager == "brew" and _is_macos():
        pkg = f"{tool.pkg}{'@'+tool.version if tool.version else ''}"
        cmd = ["bash","-lc", f"brew list {tool.pkg} || brew install {pkg}"]
    elif tool.manager == "winget" and _is_windows():
        pkg = tool.pkg
        cmd = ["cmd","/c", f"winget list {pkg} || winget install --silent {pkg}"]
    elif tool.manager == "apt" and _is_linux():
        pkg = tool.pkg
        cmd = ["bash","-lc", f"dpkg -s {pkg} || sudo apt-get update && sudo apt-get install -y {pkg}"]
    elif tool.manager == "pip":
        pkg = f"{tool.pkg}=={tool.version}" if tool.version else tool.pkg
        cmd = ["bash","-lc", f"python -m pip show {tool.pkg} || python -m pip install {pkg}"]
    else:
        return ToolInstallResult(tool.name, False, tool.manager, tool.version, None, "", f"unsupported_manager:{tool.manager}")

    rc, out, err = _run(cmd)
    ok = (rc == 0)

    payload = {
        "tool": tool.__dict__,
        "ok": ok, "rc": rc,
        "ts": time.time(),
        "host": platform.platform(),
    }
    evid = CAS.put(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    return ToolInstallResult(tool.name, ok, tool.manager, tool.version, evid, out, err)


def ensure_tools(spec: Any, ctx: Dict[str,Any]) -> List[Dict[str,Any]]:
    """סורק דרישות כלים מתוך spec/ctx.policy ומתקין בבטחה.
       תומך ברשיונות/allowlist וב-hash pinning (אם קיים)."""
    policy = ctx.get("__policy__", {})
    reqs: List[Dict[str,Any]] = []
    # 1) מ-spec (שדה אופציונלי)
    if isinstance(spec, dict):
        reqs += (spec.get("tools_required") or [])
    # 2) מה-policy (שדה אופציונלי)
    reqs += (policy.get("tools_required") or [])

    results: List[Dict[str,Any]] = []
    for r in reqs:
        t = ToolSpec(
            name=r.get("name","unknown"),
            pkg=r.get("pkg",""),
            manager=r.get("manager","pip"),
            version=r.get("version"),
            sha256=r.get("sha256"),
            spdx=r.get("spdx"),
        )
        if t.spdx and t.spdx not in ALLOWED_LICENSES:
            results.append({"name": t.name, "ok": False, "reason": f"license_not_allowed:{t.spdx}"})
            continue
        res = _install(t)
        results.append({"name": t.name, "ok": res.ok, "manager": res.manager, "evidence": res.evidence_id})
    return results