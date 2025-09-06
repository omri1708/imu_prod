# server/archive_api.py
# Export/Import ZIP: policies + public keys + provenance envelopes + artifact index + CAS.
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path
import json, os, time, shutil

router = APIRouter(prefix="/archive", tags=["archive"])

ROOT = Path(".")
CAS_DIRS = ["cas", ".imu/provenance", ".imu/artifacts"]

def _add_if_exists(z: ZipFile, rel: str):
    p = ROOT / rel
    if not p.exists(): return
    if p.is_file():
        z.write(p, arcname=rel)
    else:
        for sub in p.rglob("*"):
            if sub.is_file():
                z.write(sub, arcname=str(sub.relative_to(ROOT)))

@router.get("/export")
def export_zip(include_private: bool = False):
    buf = BytesIO()
    with ZipFile(buf, "w", compression=ZIP_DEFLATED) as z:
        # policies
        _add_if_exists(z, "security/policy_rules.yaml")
        # public keys only
        _add_if_exists(z, ".imu/keys/pub")
        if include_private:
            _add_if_exists(z, ".imu/keys/priv")
        # provenance/ CAS/ artifact-index
        for d in CAS_DIRS:
            _add_if_exists(z, d)
        # metrics snapshot
        snap = {"ts": time.time()}
        z.writestr("snapshots/README.txt", "IMU export snapshot\n")
        z.writestr("snapshots/when.json", json.dumps(snap, ensure_ascii=False, indent=2))
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition":"attachment; filename=imu_export.zip"})

class ImportZip(BaseModel):
    archive_path: str

@router.post("/import")
def import_zip(req: ImportZip):
    p = Path(req.archive_path)
    if not p.exists():
        raise HTTPException(404, "archive not found")
    with ZipFile(p, "r") as z:
        for name in z.namelist():
            if name.endswith("/"):  # directory
                continue
            # אל תדרוס מפתחות priv אם קיימים, אלא אם נעול (כאן נזהר—נדלג כברירת מחדל)
            if name.startswith(".imu/keys/priv/") and (ROOT / name).exists():
                continue
            (ROOT / name).parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, open(ROOT / name, "wb") as dst:
                shutil.copyfileobj(src, dst)
    return {"ok": True}