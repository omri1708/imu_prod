# server/deps/admin_auth.py
import os
from fastapi import Header, HTTPException

def require_admin(x_admin_token: str | None = Header(default=None, alias="X-Admin-Token")) -> None:
    expected = os.getenv("IMU_ADMIN_TOKEN", "")
    if not expected or x_admin_token != expected:
        raise HTTPException(status_code=401, detail="admin token required")
