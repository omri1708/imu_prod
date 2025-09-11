# imu_repo/tests/test_stage93_device_caps.py
from __future__ import annotations
from grounded.claims import current
from engine.user_scope import user_scope
from engine.config import load_config, save_config
from device.caps_device import perm_get_capability, perm_set_capability, sensor_read_capability

def _cfg():
    cfg = load_config()
    cfg["guard"] = {"min_trust": 0.0, "max_age_s": 3600.0, "min_count": 0, "required_kinds": []}
    cfg["phi"] = {"max_allowed": 200.0, "per_capability_cost": {
        "device.permission.get": 0.2, "device.permission.set":0.3, "device.sensor.read":0.5}}
    save_config(cfg)

def test_permissions_and_sensor_requests():
    _cfg()
    current().reset()
    with user_scope("lior"):
        getc = perm_get_capability("lior")
        setc = perm_set_capability("lior")
        sens = sensor_read_capability("lior")

        out0 = getc.sync({})
        assert "perm:" in out0["text"]

        # בלי הרשאה — geolocation יחסם
        out1 = sens.sync({"kind":"geolocation"})
        assert "permission_denied" in out1["text"]

        # נעניק geolocation בלבד
        out2 = setc.sync({"geolocation": True})
        assert out2["text"] == "perm:ok"

        out3 = getc.sync({})
        assert "True" in out3["text"]  # geolocation=True

        # כעת בקשת חיישן תחזיר "ACTION" לשכבת UI (שמבצעת בדפדפן בפועל)
        out4 = sens.sync({"kind":"geolocation"})
        assert "[ACTION]" in out4["text"]

        # Evidences קיימים
        kinds = {e["kind"] for e in current().snapshot()}
        assert "device_policy_update" in kinds
        assert "device_sensor_request" in kinds or "device_perm_block" in kinds

def run():
    test_permissions_and_sensor_requests()
    print("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())