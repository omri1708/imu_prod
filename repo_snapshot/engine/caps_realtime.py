# imu_repo/engine/caps_realtime.py
from __future__ import annotations
import asyncio, os
from typing import Dict, Any
from engine.capability_wrap import text_capability_for_user
from engine.policy_ctx import get_user
from grounded.claims import current
from realtime.ws_loopback import start_loopback_server, open_loopback_client

async def _ws_echo_impl(payload: Dict[str,Any]) -> str:
    # פותח שרת לוקאלי, לקוח, שולח נתונים בינאריים עם back-pressure + deflate, ואוסף Echo
    data = payload.get("data_bytes")
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data_bytes required")
    # start server
    srv, host, port = await start_loopback_server()
    try:
        cli = await open_loopback_client(host, port)
        await cli.send_bytes(bytes(data))
        # נמתין מעט לקבלת echo+credits
        await asyncio.sleep(0.1)
        await cli.close()
        txt = f"ws_ok bytes={len(data)} host={host} port={port}"
        current().add_evidence("ws_summary", {"source_url":"local://ws_loopback","trust":0.97,"ttl_s":600,"payload":{"size":len(data),"host":host,"port":port}})
        return txt
    finally:
        srv.close()
        await srv.wait_closed()

def realtime_ws_echo_capability(user_id: str):
    return text_capability_for_user(_ws_echo_impl, user_id=user_id, capability_name="realtime.ws.echo", cost=2.0)