# imu_repo/device/policy.py
from __future__ import annotations
from typing import Dict, Any
from dataclasses import dataclass, field
from grounded.claims import current
from engine.kvstore import load_kv, save_kv
from engine.policy_ctx import get_user

@dataclass
class PermissionState:
    geolocation: bool = False
    microphone: bool = False
    camera: bool = False

def _key(user: str) -> str:
    return f"device_policy/{user}"

def get_policy(user: str | None = None) -> PermissionState:
    u = user or get_user() or "anon"
    kv = load_kv(_key(u)) or {}
    return PermissionState(**{**PermissionState().__dict__, **kv})

def set_policy(state: PermissionState, user: str | None = None) -> None:
    u = user or get_user() or "anon"
    save_kv(_key(u), {
        "geolocation": bool(state.geolocation),
        "microphone": bool(state.microphone),
        "camera": bool(state.camera),
    })
    current().add_evidence("device_policy_update", {
        "source_url":"imu://device/policy","trust":0.98,"ttl_s":3600,
        "payload":{"user": u, **state.__dict__}
    })