# adapters/docker_sign.py
# Sign docker images with cosign if available; otherwise return resource_required + install command.
from __future__ import annotations
import shutil, subprocess, json
from typing import Dict, Any

def have(x:str)->bool: return shutil.which(x) is not None

def sign_with_cosign(image: str, key_file: str | None = None) -> Dict[str, Any]:
    if not have("cosign"):
        # request-and-continue response (call /capabilities/request on client)
        return {"ok": False, "resource_required": "cosign"}
    cmd = ["cosign","sign"]
    if key_file:
        cmd += ["--key", key_file]
    cmd += [image]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return {"ok": True, "stdout": out.stdout[-800:]}
    except subprocess.CalledProcessError as e:
        return {"ok": False, "stderr": (e.stdout or "") + (e.stderr or "")}