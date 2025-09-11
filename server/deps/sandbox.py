# server/deps/sandbox.py
import os, shutil
from fastapi import HTTPException

def require_sandbox_ready() -> None:
    name = os.getenv("IMU_SANDBOX", "").strip().lower()
    if name not in ("bwrap", "firejail"):
        raise HTTPException(503, "Sandbox not configured: set IMU_SANDBOX=bwrap")
    if shutil.which(name) is None:
        raise HTTPException(503, f"Sandbox binary '{name}' not found in PATH")
