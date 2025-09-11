# imu_repo/device/caps_device.py
from __future__ import annotations
from typing import Dict, Any
from grounded.claims import current
from engine.capability_wrap import text_capability_for_user
from engine.policy_ctx import get_user
from device.policy import get_policy, set_policy, PermissionState

class ResourceRequired(Exception): ...

async def _perm_get_impl(payload: Dict[str, Any]) -> str:
    uid = get_user() or "anon"
    st = get_policy(uid)
    current().add_evidence("device_perm_get", {
        "source_url":"imu://device/policy","trust":0.96,"ttl_s":600,
        "payload":{"user": uid, **st.__dict__}
    })
    return f"perm:{st.__dict__}"

async def _perm_set_impl(payload: Dict[str, Any]) -> str:
    uid = get_user() or "anon"
    p = PermissionState(
        geolocation=bool(payload.get("geolocation", False)),
        microphone=bool(payload.get("microphone", False)),
        camera=bool(payload.get("camera", False)),
    )
    set_policy(p, uid)
    return "perm:ok"

async def _sensor_read_impl(payload: Dict[str, Any]) -> str:
    """
    payload: {"kind": "geolocation"|"microphone"|"camera"}
    הערה: בצד שרת אין בפועל גישה לחומרה — תוחזר בקשת משאב מפורשת.
    בצד UI/דפדפן — הקריאה מתבצעת דרך JS (ראה ui/render _base_js)
    """
    uid = get_user() or "anon"
    kind = str(payload.get("kind","")).lower()
    st = get_policy(uid)

    if kind == "geolocation":
        if not st.geolocation:
            current().add_evidence("device_perm_block", {"source_url":"imu://device/policy","trust":0.7,"ttl_s":300,
                "payload":{"user": uid, "kind":kind}})
            return "[FALLBACK] permission_denied: geolocation"
        # אין ספק OS צד-שרת → דרוש דפדפן/ספק
        current().add_evidence("device_sensor_request", {
            "source_url":"imu://device/sensor","trust":0.9,"ttl_s":60,
            "payload":{"user":uid,"kind":kind,"provider":"browser:navigator.geolocation"}
        })
        return "[ACTION] use_ui_button('sensor:geo')"
    elif kind in ("microphone","camera"):
        allowed = (st.microphone if kind=="microphone" else st.camera)
        if not allowed:
            current().add_evidence("device_perm_block", {"source_url":"imu://device/policy","trust":0.7,"ttl_s":300,
                "payload":{"user": uid, "kind":kind}})
            return f"[FALLBACK] permission_denied: {kind}"
        current().add_evidence("device_sensor_request", {
            "source_url":"imu://device/sensor","trust":0.9,"ttl_s":60,
            "payload":{"user":uid,"kind":kind,"provider":"browser:mediaDevices.getUserMedia"}
        })
        action = "perm:microphone" if kind=="microphone" else "perm:camera"
        return f"[ACTION] use_ui_button('{action}')"
    else:
        return f"[FALLBACK] unknown_sensor:{kind}"

def perm_get_capability(user_id: str):
    return text_capability_for_user(_perm_get_impl, user_id=user_id, capability_name="device.permission.get", cost=0.2)

def perm_set_capability(user_id: str):
    return text_capability_for_user(_perm_set_impl, user_id=user_id, capability_name="device.permission.set", cost=0.3)

def sensor_read_capability(user_id: str):
    return text_capability_for_user(_sensor_read_impl, user_id=user_id, capability_name="device.sensor.read", cost=0.5)