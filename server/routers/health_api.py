# server/routers/health_api.py
import os, shutil
from fastapi import APIRouter
from engine.blueprints.registry import list_blueprints

router = APIRouter()

@router.get("/healthz")
def healthz():
    # Registry
    bps = list_blueprints()
    assert len(bps) > 0

    # Sandbox
    name = os.getenv("IMU_SANDBOX", "").strip().lower()
    assert name in ("bwrap", "firejail") and shutil.which(name) is not None

    return {"ok": True, "blueprints": len(bps), "sandbox": name}
