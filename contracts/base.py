# contracts/base.py
from __future__ import annotations
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional, Sequence

class ContractError(Exception):
    """חוזה הופר – שימוש לא בטוח/לא מותר/חסר קדם־תנאי."""
    pass

class ResourceRequired(Exception):
    """נדרש מנוע/כלי/ספרייה חיצונית – מדווחים בדיוק מה חסר ואיך משיגים."""
    def __init__(self, resource: str, how_to_install: str):
        super().__init__(f"resource_required: {resource} | how_to_install: {how_to_install}")
        self.resource = resource
        self.how_to_install = how_to_install

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
        raise ContractError(f"command_failed: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return proc

@dataclass
class Artifact:
    path: str
    kind: str  # e.g., 'apk', 'aab', 'ipa', 'unity-bundle', 'docker-image', 'k8s-release', 'ptx', 'bin'
    provenance_sha256: Optional[str] = None
    metadata: Optional[dict] = None