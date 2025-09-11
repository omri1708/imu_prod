# imu_repo/exec/detect.py
from __future__ import annotations
import shutil, subprocess

def _probe(cmd: str, args: list[str]) -> str|None:
    exe = shutil.which(cmd)
    if not exe: return None
    try:
        out = subprocess.check_output([exe, *args], stderr=subprocess.STDOUT, timeout=4).decode(errors="ignore")
        return out.strip().splitlines()[0][:200]
    except Exception:
        return cmd  # קיים, גרסה לא ידועה

def detect() -> dict:
    return {
        "python": _probe("python3", ["--version"]) or _probe("python", ["--version"]),
        "node":   _probe("node", ["--version"]),
        "go":     _probe("go", ["version"]),
        "javac":  _probe("javac", ["-version"]),
        "java":   _probe("java", ["-version"]),
        "dotnet": _probe("dotnet", ["--version"]),
        "g++":    _probe("g++", ["--version"]),
        "rustc":  _probe("rustc", ["--version"]),
    }