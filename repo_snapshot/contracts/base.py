# contracts/base.py
from __future__ import annotations
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any


class ContractViolation(Exception): ...

class ResourceRequired(Exception):
    def __init__(self, what: str, how_to: str, env: Optional[Dict[str, Any]]=None):
        super().__init__(f"resource_required: {what}")
        self.what, self.how_to, self.env = what, how_to, (env or {})

@dataclass
class AdapterResult:
    ok: bool
    artifact_path: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    logs: str = ""
    provenance_cid: Optional[str] = None


def require(cmd: str) -> None:
    """Fail fast if a binary is missing (no external libs)."""
    import shutil
    if shutil.which(cmd) is None:
        raise ResourceRequired(
            what=f"binary:{cmd}",
            how_to=f"Install system tool `{cmd}` and ensure it is on PATH.",
        )


def ensure_tool(name: str, how_to: str) -> str:
    """מאמת שקיים כלי מערכת נדרש ומחזיר את הנתיב אליו, אחרת זורק ResourceRequired."""
    path = shutil.which(name)
    if not path:
        raise ResourceRequired(name, how_to_install=how_to)
    return path


def run_ok(cmd: Sequence[str], cwd: Optional[str] = None, env: Optional[dict] = None) -> subprocess.CompletedProcess:
    """מריץ פקודה ומוודא קוד חזרה 0, אחרת זורק ContractError עם סטנדרט־ארור מלא."""
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ContractViolation(f"command_failed: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return proc

@dataclass
class Artifact:
    path: str
    kind: str  # e.g., 'apk', 'aab', 'ipa', 'unity-bundle', 'docker-image', 'k8s-release', 'ptx', 'bin'
    provenance_sha256: Optional[str] = None
    metadata: Optional[dict] = None