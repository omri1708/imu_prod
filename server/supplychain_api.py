# server/supplychain_api.py
# FastAPI router: חתימה keyless עם cosign (Sigstore).
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os, shutil, subprocess

router = APIRouter(prefix="/supplychain", tags=["supplychain"])

class DockerSignReq(BaseModel):
    image: str
    yes: bool = True      # --yes ללא אינטראקציה
    identity_token: Optional[str] = None  # אופציונלי (OIDC), אם נדרש

def _have(x:str)->bool:
    return shutil.which(x) is not None

@router.post("/sign/docker/keyless")
def sign_docker_keyless(req: DockerSignReq) -> Dict[str,Any]:
    if not _have("cosign"):
        # “בקשה והמשך”: החזר פקודת התקנה מדויקת
        cmd = "brew install cosign" if _have("brew") else "winget install -e --id Sigstore.cosign" if _have("winget") else "curl -sSL https://raw.githubusercontent.com/sigstore/cosign/main/install.sh | sh"
        return {"ok": False, "resource_required": "cosign", "install": cmd}
    env = os.environ.copy()
    env["COSIGN_EXPERIMENTAL"] = "1"
    args = ["cosign","sign","--keyless"]
    if req.yes:
        args.append("-y")
    if req.identity_token:
        env["SIGSTORE_ID_TOKEN"] = req.identity_token
    args.append(req.image)
    try:
        out = subprocess.run(args, capture_output=True, check=True, text=True, env=env)
        return {"ok": True, "stdout": out.stdout[-1000:]}
    except subprocess.CalledProcessError as e:
        return {"ok": False, "stderr": (e.stdout or "") + (e.stderr or "")}