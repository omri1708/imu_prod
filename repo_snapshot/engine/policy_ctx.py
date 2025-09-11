# imu_repo/engine/policy_ctx.py
from __future__ import annotations
import threading
from typing import Optional

_local = threading.local()

def set_user(user_id: str) -> None:
    _local.user_id = str(user_id)

def get_user() -> Optional[str]:
    return getattr(_local, "user_id", None)

def clear_user() -> None:
    if hasattr(_local, "user_id"):
        delattr(_local, "user_id")