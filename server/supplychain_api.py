# server/supplychain_api.py  (מלא, כולל verify)
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import os, shutil, subprocess

router = APIRouter(prefix="/supplychain", tags=["supplychain"])

def _have(x:str)->bool: return shutil.which(x) is not None

class DockerSignReq(BaseModel):
    image: str
    yes: bool = True
    identity_token: Optional[str] = None  # OIDC ID token (optional)

@router.post("/sign/docker/keyless")
def sign_docker_keyless(req: DockerSignReq) -> Dict[str,Any]:
    if not _have("cosign"):
        cmd = "brew install cosign" if _have("brew") else "winget install -e --id Sigstore.cosign" if _have("winget") else "curl -sSL https://raw.githubusercontent.com/sigstore/cosign/main/install.sh | sh"
        return {"ok": False, "resource_required": "cosign", "install": cmd}
    env = os.environ.copy()
    env["COSIGN_EXPERIMENTAL"] = "1"
    args = ["cosign","sign","--keyless"]
    if req.yes: args += ["-y"]
    if req.identity_token: env["SIGSTORE_ID_TOKEN"] = req.identity_token
    args += [req.image]
    try:
        out = subprocess.run(args, capture_output=True, check=True, text=True, env=env)
        return {"ok": True, "stdout": out.stdout[-2000:]}
    except subprocess.CalledProcessError as e:
        return {"ok": False, "stderr": (e.stdout or "") + (e.stderr or "")}

class DockerVerifyReq(BaseModel):
    image: str
    certificate_identity: Optional[str] = Field(None, description="Email/URI in Fulcio cert")
    certificate_oidc_issuer: Optional[str] = Field(None, description="OIDC issuer URL, e.g. https://token.actions.githubusercontent.com")

@router.post("/verify/docker/keyless")
def verify_docker_keyless(req: DockerVerifyReq) -> Dict[str,Any]:
    if not _have("cosign"):
        cmd = "brew install cosign" if _have("brew") else "winget install -e --id Sigstore.cosign" if _have("winget") else "curl -sSL https://raw.githubusercontent.com/sigstore/cosign/main/install.sh | sh"
        return {"ok": False, "resource_required": "cosign", "install": cmd}
    env = os.environ.copy()
    env["COSIGN_EXPERIMENTAL"] = "1"
    args = ["cosign","verify","--keyless"]
    if req.certificate_identity:
        args += ["--certificate-identity", req.certificate_identity]
    if req.certificate_oidc_issuer:
        args += ["--certificate-oidc-issuer", req.certificate_oidc_issuer]
    args += [req.image]
    try:
        out = subprocess.run(args, capture_output=True, check=True, text=True, env=env)
        return {"ok": True, "stdout": out.stdout[-2000:]}
    except subprocess.CalledProcessError as e:
        return {"ok": False, "stderr": (e.stdout or "") + (e.stderr or "")}
