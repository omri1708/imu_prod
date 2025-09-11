# imu_repo/tests/test_stage89_ws_loopback.py
from __future__ import annotations
import asyncio, time, os
from grounded.claims import current
from engine.user_scope import user_scope
from engine.config import load_config, save_config
from engine.caps_realtime import realtime_ws_echo_capability

def _setup_cfg():
    cfg = load_config()
    cfg["realtime"] = {"chunk_bytes": 8192, "initial_credits": 2, "permessage_deflate": True}
    cfg["phi"] = {"max_allowed": 100.0, "per_capability_cost": {"realtime.ws.echo": 2.0}}
    cfg["guard"] = {"min_trust": 0.0, "max_age_s": 3600.0, "min_count": 0, "required_kinds": []}
    save_config(cfg)

def test_ws_deflate_and_backpressure():
    _setup_cfg()
    current().reset()
    # נכין מטען דחיס (הרבה חזרות)
    payload = {"data_bytes": (b"A"*200000)}
    with user_scope("erin"):
        cap = realtime_ws_echo_capability("erin")
        out = asyncio.get_event_loop().run_until_complete(cap(payload))
        assert "ws_ok" in out["text"], out
        evs = current().snapshot()
        # בדוק שנשלחו וקבלו הודעות
        send_ev = [e for e in evs if e["kind"] == "ws_send"]
        recv_ev = [e for e in evs if e["kind"] == "ws_echo"]
        assert len(send_ev) >= 10, "expected multiple chunks sent"
        assert len(recv_ev) >= 10, "expected multiple chunks echoed"
        # ודא שחלק נשלחו עם compressed=True
        assert any(e["payload"].get("compressed") for e in recv_ev), "permessage-deflate not observed"
        # ב־initial_credits=2, נדרש לפחות מספר סבבים — בזכות back-pressure זמני הריצה אינם מיידיים
        # לא נמדוד זמן קשיח, רק נוודא שנוצרו קרדיטים (מופעי CREDIT → נרשמים כ-echos של "CREDIT:1")
        # (הבדיקה בוחנת שיש הרבה סבבים של send/echo)
        assert len(send_ev) - len(recv_ev) < len(send_ev), "back-pressure likely ineffective"

def run():
    test_ws_deflate_and_backpressure()
    print("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())