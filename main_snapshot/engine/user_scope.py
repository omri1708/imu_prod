# imu_repo/engine/user_scope.py
from __future__ import annotations
from contextlib import contextmanager
from engine.policy_ctx import set_user, clear_user

@contextmanager
def user_scope(user_id: str):
    set_user(user_id)
    try:
        yield
    finally:
        clear_user()