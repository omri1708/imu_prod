# imu_repo/realtime/server.py
from __future__ import annotations
import asyncio
from typing import Dict, Any, Tuple
from realtime.tcp_framed import TCPFramedServer
from realtime.ws_minimal import WSServer
from realtime.strict_sink import StrictSink, Reject

# דוגם handler שמחזיר את מה שקיבל (echo) – אבל *רק* אם ה-bundle עומד בחובת claims,
# כי StrictSink יפיל כל ניסיון לשלוח ללא claims. בכך אנו אוכפים Grounded-Strict Everywhere.

async def echo_handler(op: str, bundle: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    # ניתן להוסיף כאן לוגיקה עסקית (תיעדוף, רוטינג, רישום וכו').
    return op, bundle

async def run_tcp(host: str, port: int, policy: Dict[str, Any]):
    sink = StrictSink(policy=policy)
    srv = TCPFramedServer(host, port, handler=echo_handler, sink=sink)
    await srv.run_forever()

async def run_ws(host: str, port: int, policy: Dict[str, Any]):
    sink = StrictSink(policy=policy)
    srv = WSServer(host, port, handler=echo_handler, sink=sink)
    await srv.run_forever()