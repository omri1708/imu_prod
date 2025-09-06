# server/supplychain_index_api.py
# FastAPI router: index docker image↔digest↔envelope, and cosign attest (SBOM-like).
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os, shutil, subprocess, json

from provenance.artifact_index import INDEX, ImageRecord

router = APIRouter(prefix="/supplychain/index", tags=["supplychain-index"])

class PutReq(BaseModel):
    image: str
    digest: str
    envelope_path: Optional[str] = None
    meta: Dict[str, Any] = {}

@router.post("/put")
def put(req: PutReq):
    INDEX.put(req.image, req.digest, req.envelope_path, req.meta)
    return {"ok": True}

@router.get("/get")
def get(image: str):
    items = [i.__dict__ for i in INDEX.find(image)]
    return {"ok": True, "items": items}

def _have(x:str)->bool: return shutil.which(x) is not None

class AttestReq(BaseModel):
    image: str
    predicate_path: str    # path to SBOM/predicate file (CycloneDX, SPDX, JSON)

@router.post("/attest")
def attest(req: AttestReq):
    if not _have("cosign"):
        cmd = "brew install cosign" if _have("brew") else "winget install -e --id Sigstore.cosign" if _have("winget") else "curl -sSL https://raw.githubusercontent.com/sigstore/cosign/main/install.sh | sh"
        return {"ok": False, "resource_required": "cosign", "install": cmd}
    env = os.environ.copy()
    env["COSIGN_EXPERIMENTAL"] = "1"
    args = ["cosign","attest","--keyless","--predicate", req.predicate_path, "--type","cyclonedx", req.image]
    try:
        out = subprocess.run(args, check=True, capture_output=True, text=True, env=env)
        return {"ok": True, "stdout": out.stdout[-2000:]}
    except subprocess.CalledProcessError as e:
        return {"ok": False, "stderr": (e.stdout or "") + (e.stderr or "")}